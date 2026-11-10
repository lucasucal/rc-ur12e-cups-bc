"""Contestant policy: LeRobot ACT. Loads from /weights, then /app/ckpt."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from obs import MAX_DELTA
from rc_sdk.policy import Policy

WEIGHTS_DIR = Path(os.environ.get("RC_WEIGHTS_DIR", "/weights"))
LOCAL_CKPT = Path(os.environ.get("RC_LOCAL_CKPT", "/app/ckpt"))


def _looks_like_act(d: Path) -> bool:
    """True for a LeRobot ACT save_pretrained dir. False for dummy HF pins (tiny-gpt2)."""
    cfg_path = d / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(cfg, dict):
        return False
    if cfg.get("chunk_size") is not None or cfg.get("n_action_steps") is not None:
        return True
    return str(cfg.get("type") or "").lower() in {"act", "actpolicy"}


def _find_ckpt() -> Path:
    for d in (WEIGHTS_DIR, LOCAL_CKPT):
        if _looks_like_act(d):
            print(f"[act] checkpoint {d}", flush=True)
            return d
    raise FileNotFoundError(
        f"no ACT checkpoint in {WEIGHTS_DIR} or {LOCAL_CKPT} "
        "(runner mounts the manifest HF pin at /weights; dummy pins like tiny-gpt2 are ignored)"
    )


def _rgb(obs: dict[str, Any]) -> np.ndarray:
    raw = (obs.get("images") or {}).get("top")
    if not raw:
        return np.zeros((480, 640, 3), np.uint8)
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB"), dtype=np.uint8)


def _state(obs: dict[str, Any]) -> np.ndarray:
    p = obs["proprio"]
    return np.asarray(list(p["joint_pos"]) + [float(p["gripper"])], dtype=np.float32)


class ActPolicy(Policy):
    def load(self) -> None:
        from lerobot.policies.act import ACTPolicy

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = _find_ckpt()
        self.policy = ACTPolicy.from_pretrained(str(self.ckpt))
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor = None
        try:
            from lerobot.policies import make_pre_post_processors
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config, pretrained_path=str(self.ckpt),
            )
        except TypeError:
            try:
                from lerobot.policies import make_pre_post_processors
                stats = getattr(self.policy, "dataset_stats", None)
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    self.policy.config, dataset_stats=stats,
                )
            except Exception:
                self.preprocessor = self.postprocessor = None
        except Exception:
            self.preprocessor = self.postprocessor = None
        self.policy.reset()
        self.memo: tuple[tuple[Any, Any], dict] | None = None
        self.episode: Any = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        key = (obs["episode_id"], obs["step"])
        if self.memo and self.memo[0] == key:
            return self.memo[1]
        if obs["episode_id"] != self.episode or obs["step"] == 0:
            self.episode = obs["episode_id"]
            self.policy.reset()
        rgb = _rgb(obs)
        img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        state = torch.from_numpy(_state(obs)).unsqueeze(0).to(self.device)
        batch: dict[str, Any] = {
            "observation.images.top": img,
            "observation.state": state,
        }
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        if self.postprocessor is not None:
            try:
                action = self.postprocessor(action)
            except Exception:
                pass
        vec = action.detach().cpu().numpy().reshape(-1)
        values = [float(np.clip(v, -MAX_DELTA, MAX_DELTA)) for v in vec[:6]]
        gripper = float(np.clip(vec[6] if len(vec) > 6 else 0.5, 0.0, 1.0))
        out = {"type": "joint_delta", "values": values, "gripper": gripper}
        self.memo = (key, out)
        return out
