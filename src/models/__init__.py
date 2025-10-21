"""
Models package for UniMatch-Clip framework

This package contains all model definitions including domain adapters,
fuzzy attention mechanisms, and difficulty-aware processing components.
"""

from .adapters import BaseDomainAdapter, DomainType, AdapterRegistry
from .vision_adapter import VisionAdapter
from .nlp_adapter import NLPAdapter
from .security_adapter import SecurityAdapter
from .medical_adapter import MedicalAdapter

__all__ = [
    "BaseDomainAdapter",
    "DomainType",
    "AdapterRegistry",
    "VisionAdapter",
    "NLPAdapter",
    "SecurityAdapter",
    "MedicalAdapter"
]