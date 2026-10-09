"""
Architecture registry.

Every HCNN variant lives in its own package under :mod:`hcnn.architectures` and
registers itself here with an :class:`ArchitectureSpec`. The registry is what lets
the rest of the code base treat architectures generically:

- ``hcnn.build_model("lform", n_obs_vars=3, n_hid_vars=10)`` builds any variant by name;
- the contract tests in ``tests/test_architecture_contract.py`` run against every
  registered architecture, so a new one is checked automatically;
- tutorials / benchmarks can loop over :func:`list_architectures`.

Registering is one call in the architecture's ``__init__.py``::

    register_architecture(ArchitectureSpec(
        name="vanilla",
        model_cls=Vanilla_Model,
        cell_cls=VanillaHCNNCell,
        ensemble_cls=VanillaHCNNEnsemble,
        summary="s_{t+1} = A tanh(s_t - C^T (y_hat_t - y_t))",
    ))
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from .base import BaseHCNNCell, BaseHCNNModel


@dataclass(frozen=True)
class ArchitectureSpec:
    """Everything the framework needs to know about one HCNN architecture.

    Attributes
    ----------
    name : str
        Short, lowercase identifier. By convention it equals the package name
        under ``hcnn/architectures/`` and the commit scope for changes to it.
    model_cls : type
        :class:`~hcnn.core.base.BaseHCNNModel` subclass. Must be constructible
        as ``model_cls(n_obs_vars=..., n_hid_vars=...)`` (all other arguments
        have defaults).
    cell_cls : type
        :class:`~hcnn.core.base.BaseHCNNCell` subclass implementing the recurrence.
    ensemble_cls : type, optional
        Ensemble wrapper (usually a thin :class:`~hcnn.core.ensembles.base.HCNNEnsemble`
        subclass). ``None`` if the architecture has no ensemble yet.
    summary : str
        One-line description of the transition, shown by :func:`describe_architectures`.
    maintainers : tuple of str
        GitHub handles of the people who own this architecture.
    """

    name: str
    model_cls: Type[BaseHCNNModel]
    cell_cls: Type[BaseHCNNCell]
    ensemble_cls: Optional[type] = None
    summary: str = ""
    maintainers: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.name or self.name != self.name.lower() or " " in self.name:
            raise ValueError(f"Architecture name must be lowercase without spaces, got {self.name!r}")
        if not issubclass(self.model_cls, BaseHCNNModel):
            raise TypeError(f"{self.model_cls.__name__} must subclass BaseHCNNModel")
        if not issubclass(self.cell_cls, BaseHCNNCell):
            raise TypeError(f"{self.cell_cls.__name__} must subclass BaseHCNNCell")


_REGISTRY: Dict[str, ArchitectureSpec] = {}


def register_architecture(spec: ArchitectureSpec) -> ArchitectureSpec:
    """Add ``spec`` to the registry. Re-registering the identical spec is a no-op."""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"Architecture {spec.name!r} is already registered by "
            f"{existing.model_cls.__module__}.{existing.model_cls.__name__}"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_architecture(name: str) -> ArchitectureSpec:
    """Look up a registered architecture by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown architecture {name!r}. Registered: {', '.join(list_architectures())}"
        ) from None


def list_architectures() -> List[str]:
    """Names of all registered architectures, sorted."""
    return sorted(_REGISTRY)


def build_model(name: str, **model_kwargs) -> BaseHCNNModel:
    """Instantiate the model of architecture ``name`` with ``model_kwargs``."""
    return get_architecture(name).model_cls(**model_kwargs)


def build_ensemble(name: str, n_ensemble: int, **model_kwargs):
    """Instantiate an ensemble of ``n_ensemble`` members of architecture ``name``."""
    spec = get_architecture(name)
    if spec.ensemble_cls is None:
        raise ValueError(f"Architecture {name!r} does not provide an ensemble")
    return spec.ensemble_cls(n_ensemble=n_ensemble, **model_kwargs)


def describe_architectures() -> str:
    """Human-readable table of the registered architectures."""
    width = max((len(n) for n in _REGISTRY), default=4)
    return "\n".join(
        f"{spec.name:<{width}}  {spec.model_cls.__name__:<14} {spec.summary}"
        for spec in (_REGISTRY[n] for n in list_architectures())
    )
