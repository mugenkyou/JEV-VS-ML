"""Classical ML Subsystem: Registries, Hardware Backend Routing, and Training Orchestration."""
from .registry import candidates
from .backends import route, probe_cuml
from .training import fit_candidate, finish_result, train_job, train_lane

__all__ = [
    "candidates",
    "route",
    "probe_cuml",
    "fit_candidate",
    "finish_result",
    "train_job",
    "train_lane"
]
