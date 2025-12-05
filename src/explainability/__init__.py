"""
Explainability module for DeepFake Detection.

This package provides tools for explaining decisions made by
linear probing classifiers trained on DINOv2 and OpenCLIP embeddings.
"""

from .feature_importance import FeatureImportanceAnalyzer
from .shap_explainer import SHAPExplainer
from .embedding_visualizer import EmbeddingVisualizer

__all__ = [
    'FeatureImportanceAnalyzer',
    'SHAPExplainer',
    'EmbeddingVisualizer',
]
