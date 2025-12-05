"""
DeepFake Detection with Explainable AI.

This package provides tools for:
- Feature importance analysis for linear probing classifiers
- SHAP-based explanations for model predictions
- Embedding space visualization and analysis
"""

__version__ = '1.0.0'

from .explainability import (
    FeatureImportanceAnalyzer,
    SHAPExplainer,
    EmbeddingVisualizer
)

from .utils import (
    DataLoader,
    load_embeddings,
    ReportGenerator
)

__all__ = [
    'FeatureImportanceAnalyzer',
    'SHAPExplainer',
    'EmbeddingVisualizer',
    'DataLoader',
    'load_embeddings',
    'ReportGenerator',
]
