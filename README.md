# UR12e cup-family BC policy

Behavior-cloned visuomotor policy for the Reality Check UR12e + 2F-140 cell
(schema v2). Weights live on Hugging Face (`manifest.json`); this repo is the
Docker build context only.

- `load()` reads `/weights/model.pt`
- `act()` maps overhead JPEG + proprio → 6 `joint_delta` + gripper
