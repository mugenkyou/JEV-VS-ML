"""Data Subsystem: Ingestion, Snapshots, Holdouts, and Stratified Splitting."""
from .datasets import download, hf_frames, dataset_spec, prepare_data
from .sampling import cap_indices
from .splits import make_holdout, make_split, pilot_split

__all__ = [
    "download", "hf_frames", "dataset_spec", "prepare_data",
    "cap_indices", "make_holdout", "make_split", "pilot_split"
]
