"""
HCNN ensemble implementations.

A single generic ensemble base (:class:`_HCNNEnsembleBase`) wraps N independent
member models and provides:

- aggregation over the *member* axis: ``mean``, ``median``, and a real
  ``weighted_mean`` (weights normalized to a convex combination; works for every
  variant, not just Vanilla, and broadcasts to tensors of any rank);
- uncertainty quantification for every variant via
  :meth:`predict_with_uncertainty` (mean, variance, per-member predictions, and a
  scalar ``model_agreement``);
- reproducible member diversity via an optional ``seed`` (member i is seeded with
  ``seed + i``), so members are guaranteed to differ yet the ensemble is
  reproducible.

The four public classes are thin subclasses that fix the member model type.
Members are held in an ``nn.ModuleDict`` so all parameters are registered and
``.to(device)`` / ``state_dict()`` behave correctly.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from datetime import datetime

from ..base import BaseEnsemble
from ..models.hcnn_models import Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model


class _HCNNEnsembleBase(BaseEnsemble):
    """Generic ensemble of HCNN member models (see module docstring)."""

    #: Member model class; set by each concrete subclass.
    model_cls = None
    #: Short label used in the generated name / ensemble_type.
    variant_label = "hcnn"

    def __init__(self, n_ensemble: int, seed: Optional[int] = None, **model_kwargs):
        super().__init__(n_ensemble)
        if self.model_cls is None:
            raise TypeError("_HCNNEnsembleBase is abstract; use a concrete subclass.")
        self.model_kwargs = model_kwargs
        self.seed = seed

        # Build members. Seeding per member (seed + i) guarantees they differ while
        # keeping the whole ensemble reproducible.
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            if seed is not None:
                torch.manual_seed(seed + i)
            self.models[f"model_{i}"] = self.model_cls(**model_kwargs)

        self.name = self._generate_ensemble_name()

    # -- introspection ----------------------------------------------------
    @property
    def ensemble_type(self) -> str:
        return f"{self.variant_label}_hcnn_ensemble"

    def _generate_ensemble_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.__class__.__name__}_n{self.n_ensemble}_{timestamp}"

    def get_model(self, index: int):
        """Return member ``index``."""
        return self.models[f"model_{index}"]

    def get_all_models(self) -> List[nn.Module]:
        """Return all member models as a list."""
        return list(self.models.values())

    # -- member rollout ---------------------------------------------------
    def _run_members(self, data_window, forecast_horizon, externals, future_externals):
        return [
            m(
                data_window,
                forecast_horizon=forecast_horizon,
                externals=externals,
                future_externals=future_externals,
            )
            for m in self.models.values()
        ]

    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean",
        weights: Optional[torch.Tensor] = None,
    ):
        """Run all members and aggregate their outputs over the member axis."""
        outputs = self._run_members(data_window, forecast_horizon, externals, future_externals)
        return self.aggregate_outputs(outputs, aggregation_method, weights)

    # -- aggregation ------------------------------------------------------
    @staticmethod
    def _normalized_weights(n: int, weights, device, dtype) -> torch.Tensor:
        if weights is None:
            return torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        w = torch.as_tensor(weights, device=device, dtype=dtype)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {tuple(w.shape)}")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        return w / total  # normalize to a convex combination

    def _reduce(self, stacked: torch.Tensor, method: str, weights) -> torch.Tensor:
        """Reduce a ``(n_members, ...)`` stack over the member axis (dim 0)."""
        if method == "mean":
            return stacked.mean(dim=0)
        if method == "median":
            return stacked.median(dim=0)[0]
        if method == "weighted_mean":
            w = self._normalized_weights(stacked.shape[0], weights, stacked.device, stacked.dtype)
            view = [-1] + [1] * (stacked.dim() - 1)  # broadcast over any rank
            return (stacked * w.view(*view)).sum(dim=0)
        raise ValueError(f"Unknown aggregation method: {method}")

    def aggregate_outputs(self, outputs: List, method: str = "mean", weights=None):
        """
        Aggregate a list of member outputs (same named-tuple type) field-by-field
        over the member axis. Fields that are ``None`` for any member (e.g.
        forecasts when no horizon was requested) stay ``None``.
        """
        out_type = type(outputs[0])
        reduced = {}
        for field in outputs[0]._fields:
            vals = [getattr(o, field) for o in outputs]
            if any(v is None for v in vals):
                reduced[field] = None
            else:
                reduced[field] = self._reduce(torch.stack(vals, dim=0), method, weights)
        return out_type(**reduced)

    # -- uncertainty quantification --------------------------------------
    def predict_with_uncertainty(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        return_individual: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with ensemble uncertainty (available for every variant).

        Returns a dict with ``predictions_mean``/``predictions_var`` (over the
        calibration expectations), ``forecasts_mean``/``forecasts_var`` when a
        horizon is given, and a scalar ``model_agreement`` in (0, 1] (1 = members
        agree perfectly). Variance uses the population estimator (unbiased=False)
        so a single-member ensemble yields 0 rather than NaN. With
        ``return_individual=True`` the per-member expectations (and forecasts) are
        included.
        """
        outputs = self._run_members(data_window, forecast_horizon, externals, future_externals)
        exp = torch.stack([o.expectations for o in outputs], dim=0)  # (n, B, T, obs)

        mean = exp.mean(dim=0)
        var = exp.var(dim=0, unbiased=False)
        # Agreement: shrink with dispersion; 1 when members are identical.
        mean_std = var.sqrt().mean()
        model_agreement = (1.0 / (1.0 + mean_std)).item()

        result: Dict[str, object] = {
            "predictions_mean": mean,
            "predictions_var": var,
            "model_agreement": model_agreement,
        }
        if outputs[0].forecasts is not None:
            fc = torch.stack([o.forecasts for o in outputs], dim=0)
            result["forecasts_mean"] = fc.mean(dim=0)
            result["forecasts_var"] = fc.var(dim=0, unbiased=False)
            if return_individual:
                result["individual_forecasts"] = fc
        if return_individual:
            result["individual_predictions"] = exp
        return result

    # Backward-compatible alias for the old method name.
    def get_ensemble_predictions_variance(self, data_window, forecast_horizon=None,
                                          externals=None, future_externals=None):
        r = self.predict_with_uncertainty(data_window, forecast_horizon, externals, future_externals)
        out = {"expectations_mean": r["predictions_mean"], "expectations_var": r["predictions_var"]}
        if "forecasts_mean" in r:
            out["forecasts_mean"] = r["forecasts_mean"]
            out["forecasts_var"] = r["forecasts_var"]
        return out


class VanillaHCNNEnsemble(_HCNNEnsembleBase):
    """Ensemble of Vanilla HCNN models."""
    model_cls = Vanilla_Model
    variant_label = "vanilla"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         n_ext_vars=n_ext_vars)


class PTFHCNNEnsemble(_HCNNEnsembleBase):
    """Ensemble of Partial Teacher Forcing HCNN models."""
    model_cls = PTF_Model
    variant_label = "ptf"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), target_prob=0.25,
                 dropout_strategy="adaptive", dropout_params=None, n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         target_prob=target_prob, dropout_strategy=dropout_strategy,
                         dropout_params=dropout_params, n_ext_vars=n_ext_vars)

    def update_dropout_epoch(self, epoch: int):
        """Advance the dropout schedule of every member (for PTF training loops)."""
        for m in self.models.values():
            m.update_dropout_epoch(epoch)


class LFormHCNNEnsemble(_HCNNEnsembleBase):
    """Ensemble of LSTM-formulation HCNN models."""
    model_cls = LForm_Model
    variant_label = "lform"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), init_diag=1.0, n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         init_diag=init_diag, n_ext_vars=n_ext_vars)


class LSpaHCNNEnsemble(_HCNNEnsembleBase):
    """Ensemble of Large-Sparse HCNN models."""
    model_cls = LSpa_Model
    variant_label = "lspa"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), bias=False,
                 mask_type="random_block", sparsity_ratio=0.25, n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         bias=bias, mask_type=mask_type, sparsity_ratio=sparsity_ratio,
                         n_ext_vars=n_ext_vars)

    def get_ensemble_sparsity_info(self) -> Dict[str, Dict]:
        """Sparsity statistics for every member."""
        return {f"model_{i}": m.get_sparsity_info() for i, m in enumerate(self.models.values())}
