"""Defaults shared across the HCNN framework (one place, so base classes and models agree)."""

from typing import Tuple

# Weight-initialisation range: the default of the base classes, the initial state, and every
# architecture's cell and model, so architectures compared at their defaults start at the same
# scale. (The *HCNNEnsemble constructors still default to (-0.75, 0.75).)
DEFAULT_INIT_RANGE: Tuple[float, float] = (-0.5, 0.5)
