"""
Feature Importance Analyzer for Linear Probing Classifiers.

This module provides methods to analyze which features (embedding dimensions)
are most important for the classification decision between real and fake images.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for linear probing classifiers.
    
    This class provides multiple methods to understand which embedding dimensions
    are most discriminative for distinguishing real from fake images.
    
    Attributes:
        classifier: The trained linear classifier (e.g., LogisticRegression)
        feature_names: Optional names for each embedding dimension
        scaler: Optional StandardScaler used during training
    """
    
    def __init__(
        self,
        classifier: Any = None,
        feature_names: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None
    ):
        """
        Initialize the Feature Importance Analyzer.
        
        Args:
            classifier: A trained sklearn classifier with coef_ attribute
            feature_names: Optional list of names for each feature dimension
            scaler: Optional StandardScaler if embeddings were normalized
        """
        self.classifier = classifier
        self.feature_names = feature_names
        self.scaler = scaler
        
    def fit_classifier(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
        **kwargs
    ) -> 'FeatureImportanceAnalyzer':
        """
        Fit a logistic regression classifier on the embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: Array of shape (n_samples,) with binary labels (0=real, 1=fake)
            normalize: Whether to standardize embeddings before fitting
            **kwargs: Additional arguments passed to LogisticRegression
            
        Returns:
            self: The fitted analyzer
        """
        if normalize:
            self.scaler = StandardScaler()
            embeddings = self.scaler.fit_transform(embeddings)
            
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        default_params.update(kwargs)
        
        self.classifier = LogisticRegression(**default_params)
        self.classifier.fit(embeddings, labels)
        
        if self.feature_names is None:
            self.feature_names = [f"dim_{i}" for i in range(embeddings.shape[1])]
            
        return self
    
    def get_coefficient_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance based on classifier coefficients.
        
        For logistic regression, the absolute value of coefficients indicates
        how much each feature contributes to the decision boundary.
        
        Returns:
            Tuple of (importance_scores, sorted_indices)
        """
        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")
            
        coefficients = self.classifier.coef_.flatten()
        importance = np.abs(coefficients)
        sorted_indices = np.argsort(importance)[::-1]
        
        return importance, sorted_indices
    
    def get_directional_importance(self) -> Dict[str, np.ndarray]:
        """
        Get directional feature importance showing which class each feature favors.
        
        Positive coefficients indicate features that favor the 'fake' class,
        negative coefficients favor the 'real' class.
        
        Returns:
            Dictionary with 'fake_indicators', 'real_indicators', and 'coefficients'
        """
        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")
            
        coefficients = self.classifier.coef_.flatten()
        
        fake_mask = coefficients > 0
        real_mask = coefficients < 0
        
        return {
            'coefficients': coefficients,
            'fake_indicators': np.where(fake_mask)[0],
            'real_indicators': np.where(real_mask)[0],
            'fake_importance': coefficients[fake_mask],
            'real_importance': np.abs(coefficients[real_mask])
        }
    
    def analyze_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze which features contribute most to the prediction for a single embedding.
        
        Args:
            embedding: A single embedding vector of shape (n_features,) or (1, n_features)
            top_k: Number of top contributing features to return
            
        Returns:
            Dictionary containing prediction details and feature contributions
        """
        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")
            
        embedding = np.atleast_2d(embedding)
        
        if self.scaler is not None:
            embedding_scaled = self.scaler.transform(embedding)
        else:
            embedding_scaled = embedding
            
        # Get prediction and probability
        prediction = self.classifier.predict(embedding_scaled)[0]
        probabilities = self.classifier.predict_proba(embedding_scaled)[0]
        
        # Calculate individual feature contributions
        coefficients = self.classifier.coef_.flatten()
        contributions = embedding_scaled.flatten() * coefficients
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(contributions))[::-1][:top_k]
        
        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'probability_real': probabilities[0],
            'probability_fake': probabilities[1],
            'top_contributing_features': sorted_indices.tolist(),
            'top_contributions': contributions[sorted_indices].tolist(),
            'feature_names': [self.feature_names[i] for i in sorted_indices],
            'contribution_direction': ['fake' if c > 0 else 'real' for c in contributions[sorted_indices]]
        }
    
    def plot_feature_importance(
        self,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        title: str = "Top Feature Importance for DeepFake Detection"
    ) -> plt.Figure:
        """
        Plot the top-k most important features.
        
        Args:
            top_k: Number of top features to display
            figsize: Figure size as (width, height)
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        importance, sorted_indices = self.get_coefficient_importance()
        top_indices = sorted_indices[:top_k]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        coefficients = self.classifier.coef_.flatten()
        colors = ['#e74c3c' if coefficients[i] > 0 else '#3498db' for i in top_indices]
        
        feature_labels = [self.feature_names[i] for i in top_indices]
        
        bars = ax.barh(range(top_k), importance[top_indices], color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(feature_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title(title)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Indicates Fake'),
            Patch(facecolor='#3498db', label='Indicates Real')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig
    
    def plot_contribution_breakdown(
        self,
        embedding: np.ndarray,
        top_k: int = 15,
        figsize: Tuple[int, int] = (14, 8),
        title: str = "Feature Contribution Breakdown"
    ) -> plt.Figure:
        """
        Plot a waterfall chart showing how each feature contributes to the prediction.
        
        Args:
            embedding: A single embedding vector
            top_k: Number of top contributing features to show
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        analysis = self.analyze_embedding(embedding, top_k=top_k)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        contributions = analysis['top_contributions']
        feature_labels = analysis['feature_names']
        
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in contributions]
        
        y_pos = range(len(contributions))
        ax.barh(y_pos, contributions, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Contribution to Decision (+ = Fake, - = Real)')
        ax.set_title(f"{title}\nPrediction: {analysis['prediction']} "
                    f"(Fake prob: {analysis['probability_fake']:.2%})")
        
        plt.tight_layout()
        return fig
    
    def compare_embeddings(
        self,
        real_embeddings: np.ndarray,
        fake_embeddings: np.ndarray,
        top_k: int = 20,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Compare feature distributions between real and fake images.
        
        Args:
            real_embeddings: Embeddings of real images
            fake_embeddings: Embeddings of fake images
            top_k: Number of top important features to compare
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        importance, sorted_indices = self.get_coefficient_importance()
        top_indices = sorted_indices[:top_k]
        
        fig, axes = plt.subplots(4, 5, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            
            real_values = real_embeddings[:, feat_idx]
            fake_values = fake_embeddings[:, feat_idx]
            
            ax.hist(real_values, bins=30, alpha=0.5, label='Real', color='#3498db', density=True)
            ax.hist(fake_values, bins=30, alpha=0.5, label='Fake', color='#e74c3c', density=True)
            ax.set_title(f'{self.feature_names[feat_idx]}', fontsize=8)
            ax.legend(fontsize=6)
            
        plt.suptitle('Distribution of Top Important Features: Real vs Fake', fontsize=12)
        plt.tight_layout()
        return fig
