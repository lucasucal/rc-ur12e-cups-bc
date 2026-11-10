# UR12e cup-family ACT policy

Contestant-shaped LeRobot ACT for the Reality Check UR12e + 2F-140 cell
(schema v2, `joint_delta`). This folder is the Docker build context.

Real teams put weights on a **public Hugging Face** repo and point
`manifest.json` at them. The organizer live demo also copies a local
`save_pretrained` dir into `ckpt/` (gitignored) so the image can load
without a real HF ACT pin.

- `load()` looks for a LeRobot ACT checkpoint in `/weights` first, then
  `/app/ckpt`. Dummy pins (e.g. `tiny-gpt2`) are skipped: they have a
  `config.json` but no ACT `type` / `chunk_size`.
- `act()` maps overhead JPEG + proprio → 6 `joint_delta` + gripper.

```sh
# After training (see ../../train/README.md):
cp -a /path/to/act_ckpt/. ckpt/
docker build -t rc-act .
```

`FROM rc-lerobot:train` matches the Spark train image. Raise
`RC_CONTEXT_MAX_FILE_BYTES` on the backend if `model.safetensors` is over
the default 20 MB per-file cap.
