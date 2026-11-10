"""Visuomotor BC net shared by training and the contestant policy."""
from __future__ import annotations

import torch
import torch.nn as nn

from obs import CHUNK, IMAGE_SIZE, MAX_DELTA, N_TASKS, PROPRIO_DIM  # noqa: F401

__all__ = ["CupBCNet", "IMAGE_SIZE", "MAX_DELTA"]


class CupBCNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        vis_dim = 128 * 4 * 4
        self.proprio = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        self.task_emb = nn.Embedding(N_TASKS, 32)
        self.head = nn.Sequential(
            nn.Linear(vis_dim + 128 + 32, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, CHUNK * 7),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.backbone(image), self.proprio(proprio), self.task_emb(task)], dim=-1)
        raw = self.head(x).view(-1, CHUNK, 7)
        delta = MAX_DELTA * torch.tanh(raw[..., :6])
        grip = torch.sigmoid(raw[..., 6:7])
        return torch.cat([delta, grip], dim=-1)  # B, CHUNK, 7
