"""
UniMatch-Clip: Fuzzy Attention-Based Multi-Domain Learning with Difficulty-Aware Processing

This package provides a unified framework for processing heterogeneous data
across Vision, NLP, Security, and Medical domains using fuzzy attention
mechanisms and difficulty-aware processing.
"""

__version__ = "1.0.0"
__author__ = "UniMatch-Clip Contributors"
__email__ = "your.email@domain.com"

from .models.adapters import BaseDomainAdapter, DomainType, AdapterRegistry
from .models.fuzzy_attention import *
from .models.difficulty_aware import *

__all__ = [
    "BaseDomainAdapter",
    "DomainType",
    "AdapterRegistry",
    "__version__",
    "__author__",
    "__email__"
]