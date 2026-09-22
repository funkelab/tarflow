"""Regression tests for the device-independent training helpers in tarflow/utils.py."""

from __future__ import annotations

import pytest
import torch

from tarflow.utils import CosineLRSchedule, Metrics

MIN_LR, MAX_LR = 1e-6, 1e-3
WARMUP, TOTAL = 10, 50


@pytest.fixture
def optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=123.0)


def test_cosine_lr_schedule(optimizer: torch.optim.Optimizer) -> None:
    schedule = CosineLRSchedule(optimizer, WARMUP, TOTAL, MIN_LR, MAX_LR)
    assert optimizer.param_groups[0]["lr"] == MIN_LR

    lrs = [schedule.step() for _ in range(TOTAL)]
    assert optimizer.param_groups[0]["lr"] == lrs[-1]
    warmup, decay = lrs[:WARMUP], lrs[WARMUP - 1 :]
    assert warmup == sorted(warmup)  # linear warmup ...
    assert warmup[-1] == pytest.approx(MAX_LR)  # ... to max_lr
    assert lrs[WARMUP // 2 - 1] == pytest.approx(MIN_LR + (MAX_LR - MIN_LR) / 2)
    assert decay == sorted(decay, reverse=True)  # cosine decay ...
    assert decay[-1] == pytest.approx(MIN_LR)  # ... to min_lr
    halfway = WARMUP + (TOTAL - WARMUP) // 2 - 1
    assert lrs[halfway] == pytest.approx((MIN_LR + MAX_LR) / 2)


def test_cosine_lr_schedule_state_dict(optimizer: torch.optim.Optimizer) -> None:
    """The step counter is checkpointed, so that training can be resumed."""
    schedule = CosineLRSchedule(optimizer, WARMUP, TOTAL, MIN_LR, MAX_LR)
    for _ in range(20):
        schedule.step()
    resumed = CosineLRSchedule(optimizer, WARMUP, TOTAL, MIN_LR, MAX_LR)
    resumed.load_state_dict(schedule.state_dict())
    assert resumed.step() == schedule.step()


def test_metrics() -> None:
    metrics = Metrics()
    metrics.update({"loss": torch.tensor(1.0), "lr": 0.1})
    metrics.update({"loss": 3.0})
    assert metrics.compute(None) == {"loss": 2.0, "lr": 0.1}
