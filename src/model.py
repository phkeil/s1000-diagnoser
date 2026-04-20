"""Baseline model definitions for diagnosis experiments."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Simple model configuration placeholder."""

    input_dim: int
    num_classes: int
