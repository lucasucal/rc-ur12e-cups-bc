"""Contestant policy: CNN+MLP behavior clone. Weights come from /weights."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from model import CupBCNet
from obs import CHUNK, IMAGE_SIZE, MAX_DELTA, proprio_vec, task_index
from rc_sdk.policy import Policy

WEIGHTS_DIR = Path(os.environ.get("RC_WEIGHTS_DIR", "/weights"))


def _image(obs: dict[str, Any]) -> torch.Tensor:
    raw = (obs.get("images") or {}).get("top")
    if not raw:
        arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    else:
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        arr = np.asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0)


class CupBCPolicy(Policy):
    def load(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = WEIGHTS_DIR / "model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"expected {ckpt} (HF snapshot mounted at /weights)")
        self.net = CupBCNet().to(self.device)
        blob = torch.load(ckpt, map_location=self.device, weights_only=True)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        self.net.load_state_dict(state)
        self.net.eval()
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        prop = torch.zeros(1, 20, device=self.device)
        task = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.net(dummy, prop, task)
        self.memo: tuple[tuple[Any, Any], dict] | None = None
        self.chunk: np.ndarray | None = None
        self.chunk_i = CHUNK
        self.episode: Any = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        key = (obs["episode_id"], obs["step"])
        if self.memo and self.memo[0] == key:
            return self.memo[1]
        if obs["episode_id"] != self.episode or obs["step"] == 0:
            self.episode = obs["episode_id"]
            self.chunk_i = CHUNK
        if self.chunk is None or self.chunk_i >= CHUNK:
            image = _image(obs).to(self.device)
            prop = torch.tensor([proprio_vec(obs["proprio"])], dtype=torch.float32, device=self.device)
            task = torch.tensor([task_index(obs.get("task_id"))], dtype=torch.long, device=self.device)
            with torch.no_grad():
                self.chunk = self.net(image, prop, task)[0].detach().cpu().numpy()
            self.chunk_i = 0
        out = self.chunk[self.chunk_i]
        self.chunk_i += 1
        values = [float(np.clip(v, -MAX_DELTA, MAX_DELTA)) for v in out[:6]]
        gripper = float(np.clip(out[6], 0.0, 1.0))
        action = {"type": "joint_delta", "values": values, "gripper": gripper}
        self.memo = (key, action)
        return action
