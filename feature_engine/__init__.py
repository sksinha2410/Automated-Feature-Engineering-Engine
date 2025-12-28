"""
Automated Feature Engineering Engine

A library for automated feature generation and selection for tabular data.
"""

__version__ = "0.1.0"

from .transformers import PolynomialTransformer, BinningTransformer, TargetEncoder
from .scoring import FeatureScorer
from .selection import FeatureSelector
from .pipeline import AutoFeatureEngineeringPipeline

__all__ = [
    'PolynomialTransformer',
    'BinningTransformer',
    'TargetEncoder',
    'FeatureScorer',
    'FeatureSelector',
    'AutoFeatureEngineeringPipeline'
]
