"""Observation packing shared by the contestant policy and the trainer. No torch."""
from __future__ import annotations

TASK_IDS = (
    "cup_lift_v0",
    "cup_pick_red_v0",
    "cup_place_v0",
    "cup_tip_v0",
    "cup_stack_v0",
)
UNKNOWN_TASK = len(TASK_IDS)
N_TASKS = UNKNOWN_TASK + 1
PROPRIO_DIM = 20
IMAGE_SIZE = 224
MAX_DELTA = 0.049
CHUNK = 8


def task_index(task_id: str | None) -> int:
    if task_id in TASK_IDS:
        return TASK_IDS.index(task_id)
    return UNKNOWN_TASK


def proprio_vec(proprio: dict) -> list[float]:
    return list(proprio["joint_pos"]) + list(proprio["joint_vel"]) + [
        float(proprio["gripper"]),
        *proprio["ee_pos"],
        *proprio["ee_quat"],
    ]
