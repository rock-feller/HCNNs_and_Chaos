"""Defaults shared across the HCNN framework (one place, so base classes and models agree)."""

from typing import Tuple

# Uniform weight-initialisation range: the default of the base classes, the initial state,
# and the Vanilla / PTF / template cells and models. LForm and LSpa (cells and models) and the
# *HCNNEnsemble constructors still default to (-0.75, 0.75); changing those changes results.
DEFAULT_INIT_RANGE: Tuple[float, float] = (-0.5, 0.5)
