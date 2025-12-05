"""
SHAP-based Explainer for DeepFake Detection.

This module provides SHAP (SHapley Additive exPlanations) based explanations
for understanding model predictions on image embeddings.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based explanation for linear probing classifiers.
    
    SHAP values provide a unified measure of feature importance that shows
    how each feature contributes to pushing the model output from the base
    value (average prediction) to the actual prediction.
    
    Attributes:
        classifier: The trained classifier
        explainer: The SHAP explainer object
        background_data: Background dataset for SHAP calculations
    """
    
    def __init__(
        self,
        classifier: Any = None,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the SHAP Explainer.
        
        Args:
            classifier: A trained sklearn classifier
            background_data: Background data for SHAP kernel explainer
            feature_names: Optional names for each feature dimension
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Install with: pip install shap")
            
        self.classifier = classifier
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        self._shap_values = None
        
    def fit(
        self,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100
    ) -> 'SHAPExplainer':
        """
        Initialize the SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP calculations.
                            If None, uses the stored background_data.
            n_background: Number of background samples to use (for efficiency)
            
        Returns:
            self: The fitted explainer
        """
        if background_data is not None:
            self.background_data = background_data
            
        if self.background_data is None:
            raise ValueError("Background data required for SHAP explainer")
            
        if self.classifier is None:
            raise ValueError("Classifier required for SHAP explainer")
            
        # Subsample background data for efficiency
        if len(self.background_data) > n_background:
            indices = np.random.choice(len(self.background_data), n_background, replace=False)
            background_sample = self.background_data[indices]
        else:
            background_sample = self.background_data
            
        # Create SHAP explainer for linear model
        # For linear models, we can use LinearExplainer for exact SHAP values
        if hasattr(self.classifier, 'coef_'):
            self.explainer = shap.LinearExplainer(
                self.classifier,
                background_sample,
                feature_perturbation="interventional"
            )
        else:
            # Fall back to kernel explainer for non-linear models
            self.explainer = shap.KernelExplainer(
                self.classifier.predict_proba,
                background_sample
            )
            
        if self.feature_names is None:
            self.feature_names = [f"dim_{i}" for i in range(background_sample.shape[1])]
            
        return self
    
    def explain(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate SHAP values for given embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            
        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        embeddings = np.atleast_2d(embeddings)
        self._shap_values = self.explainer.shap_values(embeddings)
        
        return self._shap_values
    
    def explain_single(
        self,
        embedding: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a single embedding.
        
        Args:
            embedding: A single embedding vector
            top_k: Number of top features to return
            
        Returns:
            Dictionary containing SHAP-based explanation
        """
        embedding = np.atleast_2d(embedding)
        shap_values = self.explain(embedding)[0]
        
        # Get prediction
        prediction = self.classifier.predict(embedding)[0]
        probabilities = self.classifier.predict_proba(embedding)[0]
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
        
        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'probability_real': float(probabilities[0]),
            'probability_fake': float(probabilities[1]),
            'base_value': float(self.explainer.expected_value),
            'shap_values': shap_values.tolist(),
            'top_features': sorted_indices.tolist(),
            'top_feature_names': [self.feature_names[i] for i in sorted_indices],
            'top_shap_values': [float(shap_values[i]) for i in sorted_indices],
            'feature_directions': ['fake' if shap_values[i] > 0 else 'real' for i in sorted_indices]
        }
    
    def plot_waterfall(
        self,
        embedding: np.ndarray,
        max_display: int = 15,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a SHAP waterfall plot for a single prediction.
        
        The waterfall plot shows how each feature pushes the prediction
        from the base value to the final output.
        
        Args:
            embedding: A single embedding vector
            max_display: Maximum number of features to display
            show: Whether to display the plot immediately
            
        Returns:
            matplotlib Figure object
        """
        embedding = np.atleast_2d(embedding)
        shap_values = self.explain(embedding)
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value,
            data=embedding[0],
            feature_names=self.feature_names
        )
        
        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=show)
        return fig
    
    def plot_beeswarm(
        self,
        embeddings: np.ndarray,
        max_display: int = 20,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a SHAP beeswarm plot showing feature importance across many samples.
        
        This plot shows the distribution of SHAP values for each feature
        across all samples, colored by feature value.
        
        Args:
            embeddings: Array of embeddings
            max_display: Maximum number of features to display
            show: Whether to display the plot immediately
            
        Returns:
            matplotlib Figure object
        """
        shap_values = self.explain(embeddings)
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(embeddings), self.explainer.expected_value),
            data=embeddings,
            feature_names=self.feature_names
        )
        
        fig = plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(explanation, max_display=max_display, show=show)
        return fig
    
    def plot_bar(
        self,
        embeddings: np.ndarray,
        max_display: int = 20,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a SHAP bar plot showing mean absolute SHAP values.
        
        Args:
            embeddings: Array of embeddings
            max_display: Maximum number of features to display
            show: Whether to display the plot immediately
            
        Returns:
            matplotlib Figure object
        """
        shap_values = self.explain(embeddings)
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(embeddings), self.explainer.expected_value),
            data=embeddings,
            feature_names=self.feature_names
        )
        
        fig = plt.figure(figsize=(12, 8))
        shap.plots.bar(explanation, max_display=max_display, show=show)
        return fig
    
    def get_global_importance(
        self,
        embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate global feature importance using SHAP values.
        
        Args:
            embeddings: Array of embeddings to compute importance over
            
        Returns:
            Dictionary with feature importance rankings
        """
        shap_values = self.explain(embeddings)
        
        # Mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        return {
            'mean_abs_shap': mean_abs_shap,
            'sorted_indices': sorted_indices,
            'sorted_feature_names': [self.feature_names[i] for i in sorted_indices],
            'sorted_importance': mean_abs_shap[sorted_indices]
        }
    
    def compare_real_vs_fake(
        self,
        real_embeddings: np.ndarray,
        fake_embeddings: np.ndarray,
        top_k: int = 15,
        figsize: tuple = (14, 6)
    ) -> plt.Figure:
        """
        Compare SHAP values between real and fake images.
        
        Args:
            real_embeddings: Embeddings of real images
            fake_embeddings: Embeddings of fake images
            top_k: Number of top features to compare
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        real_shap = self.explain(real_embeddings)
        fake_shap = self.explain(fake_embeddings)
        
        # Get top features by overall importance
        all_shap = np.vstack([real_shap, fake_shap])
        mean_abs = np.abs(all_shap).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:top_k]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Real images SHAP distribution
        real_means = real_shap[:, top_indices].mean(axis=0)
        real_stds = real_shap[:, top_indices].std(axis=0)
        
        labels = [self.feature_names[i] for i in top_indices]
        y_pos = np.arange(len(labels))
        
        axes[0].barh(y_pos, real_means, xerr=real_stds, color='#3498db', alpha=0.7)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(labels)
        axes[0].invert_yaxis()
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Mean SHAP Value')
        axes[0].set_title('Real Images - SHAP Values')
        
        # Fake images SHAP distribution
        fake_means = fake_shap[:, top_indices].mean(axis=0)
        fake_stds = fake_shap[:, top_indices].std(axis=0)
        
        axes[1].barh(y_pos, fake_means, xerr=fake_stds, color='#e74c3c', alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels)
        axes[1].invert_yaxis()
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Mean SHAP Value')
        axes[1].set_title('Fake Images - SHAP Values')
        
        plt.suptitle('SHAP Value Comparison: Real vs Fake Images', fontsize=12)
        plt.tight_layout()
        return fig
