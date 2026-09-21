"""Jev-vs-ML Benchmark Package (Protocol 3.0.1 / V3).

A modular, research-grade pipeline comparing Jev semantic LLM classification
against 11 classical machine learning pipelines across NLP and tabular tasks.
"""
from .config import configuration, ALL_DATASETS, MODELS

__version__ = "3.0.1"
__all__ = ["configuration", "ALL_DATASETS", "MODELS"]
