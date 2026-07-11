"""Shared pytest fixtures. Tests run on CPU and are seeded for determinism."""
import torch
import pytest


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
