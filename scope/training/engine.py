"""Small supervised training loop for SCOPE experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.amp import GradScaler, autocast


@dataclass
class TrainStats:
    best_acc: float
    best_epoch: int
    trainable_params: int


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item() * 100.0)


def train_one_epoch(model: nn.Module, loader, optimizer, device, scaler: GradScaler | None = None, amp: bool = True) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and amp:
            with autocast('cuda'):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_n += labels.numel()
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    total_correct, total_n = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        total_correct += int((pred == labels).sum().item())
        total_n += labels.numel()
    return 100.0 * total_correct / max(1, total_n)
