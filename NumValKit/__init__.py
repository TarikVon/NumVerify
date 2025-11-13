"""
NumValKit: A toolkit for numerical data loading, prediction, and scenario validation.
"""

__version__ = "0.1.0"

from .core.base_loader import BaseLoader
from .core.base_predictor import BasePredictor

__all__ = [
    "BaseLoader",
    "BasePredictor",
]
