"""
Report Generator for Explainable AI Analysis.

This module generates comprehensive reports summarizing the explainability
analysis of DeepFake detection models.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class ReportGenerator:
    """
    Generates comprehensive reports for explainability analysis.
    
    Creates markdown and JSON reports summarizing feature importance,
    SHAP analysis, and embedding visualizations.
    """
    
    def __init__(self, model_name: str = "DeepFake Detection Model"):
        """
        Initialize the report generator.
        
        Args:
            model_name: Name of the model being analyzed
        """
        self.model_name = model_name
        self.sections = []
        self.metadata = {
            'model_name': model_name,
            'generated_at': None,
            'version': '1.0.0'
        }
        
    def add_section(
        self,
        title: str,
        content: str,
        data: Optional[Dict[str, Any]] = None
    ) -> 'ReportGenerator':
        """
        Add a section to the report.
        
        Args:
            title: Section title
            content: Markdown-formatted content
            data: Optional structured data for the section
            
        Returns:
            self for method chaining
        """
        self.sections.append({
            'title': title,
            'content': content,
            'data': data
        })
        return self
        
    def add_model_summary(
        self,
        classifier,
        accuracy: float,
        embedding_type: str,
        n_features: int,
        n_samples: int
    ) -> 'ReportGenerator':
        """
        Add model summary section.
        
        Args:
            classifier: The trained classifier
            accuracy: Model accuracy
            embedding_type: Type of embeddings (e.g., 'DINOv2', 'OpenCLIP')
            n_features: Number of features
            n_samples: Number of training samples
            
        Returns:
            self for method chaining
        """
        content = f"""
## Model Summary

| Metric | Value |
|--------|-------|
| Embedding Type | {embedding_type} |
| Number of Features | {n_features} |
| Training Samples | {n_samples} |
| Accuracy | {accuracy:.2%} |
| Classifier Type | {type(classifier).__name__} |

The model uses a linear classifier trained on pre-extracted embeddings to distinguish 
between real and AI-generated (fake) images.
"""
        data = {
            'embedding_type': embedding_type,
            'n_features': n_features,
            'n_samples': n_samples,
            'accuracy': accuracy,
            'classifier_type': type(classifier).__name__
        }
        
        return self.add_section("Model Summary", content, data)
        
    def add_feature_importance_summary(
        self,
        top_features: List[str],
        top_importance: List[float],
        directions: List[str]
    ) -> 'ReportGenerator':
        """
        Add feature importance summary.
        
        Args:
            top_features: Names of top important features
            top_importance: Importance scores
            directions: Direction each feature points ('real' or 'fake')
            
        Returns:
            self for method chaining
        """
        rows = []
        for feat, imp, direction in zip(top_features, top_importance, directions):
            indicator = "🔴 Fake" if direction == 'fake' else "🔵 Real"
            rows.append(f"| {feat} | {imp:.4f} | {indicator} |")
            
        table = "\n".join(rows)
        
        content = f"""
## Feature Importance Analysis

The following features have the highest importance in distinguishing real from fake images:

| Feature | Importance | Indicates |
|---------|------------|-----------|
{table}

### Interpretation

- **🔴 Fake indicators**: Features that, when high, suggest the image is AI-generated
- **🔵 Real indicators**: Features that, when high, suggest the image is authentic

These features represent the embedding dimensions that the linear classifier relies on 
most heavily to make its decisions.
"""
        data = {
            'top_features': top_features,
            'top_importance': top_importance,
            'directions': directions
        }
        
        return self.add_section("Feature Importance", content, data)
        
    def add_shap_summary(
        self,
        mean_abs_shap: np.ndarray,
        top_k: int = 10
    ) -> 'ReportGenerator':
        """
        Add SHAP analysis summary.
        
        Args:
            mean_abs_shap: Mean absolute SHAP values per feature
            top_k: Number of top features to show
            
        Returns:
            self for method chaining
        """
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
        
        rows = []
        for i, idx in enumerate(sorted_indices):
            rows.append(f"| {i+1} | dim_{idx} | {mean_abs_shap[idx]:.4f} |")
            
        table = "\n".join(rows)
        
        content = f"""
## SHAP Analysis

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance 
based on game theory. They show how each feature contributes to pushing the model's prediction 
from the average to its actual output.

### Top Features by Mean |SHAP| Value

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
{table}

### Key Insights

- SHAP values are additive: the sum of SHAP values equals the difference between the 
  model's prediction and the average prediction
- Positive SHAP values push predictions toward "fake", negative toward "real"
- Features with consistently high |SHAP| values are globally important
"""
        data = {
            'mean_abs_shap': mean_abs_shap.tolist() if isinstance(mean_abs_shap, np.ndarray) else mean_abs_shap,
            'top_indices': sorted_indices.tolist()
        }
        
        return self.add_section("SHAP Analysis", content, data)
        
    def add_embedding_analysis(
        self,
        embedding_type: str,
        separation_score: float,
        cluster_quality: Optional[float] = None
    ) -> 'ReportGenerator':
        """
        Add embedding space analysis summary.
        
        Args:
            embedding_type: Type of embeddings
            separation_score: Measure of class separation
            cluster_quality: Optional cluster quality metric
            
        Returns:
            self for method chaining
        """
        content = f"""
## Embedding Space Analysis ({embedding_type})

### Class Separation

The embeddings show a separation score of **{separation_score:.4f}** between real and fake images.

"""
        if cluster_quality is not None:
            content += f"The cluster quality (silhouette score) is **{cluster_quality:.4f}**.\n\n"
            
        content += """
### Visualization Insights

The embedding visualizations (PCA, t-SNE, UMAP) reveal:

1. **Cluster Structure**: How well-separated are real and fake images in the embedding space
2. **Outliers**: Unusual samples that may be harder to classify
3. **Decision Boundaries**: Where the model draws the line between classes

A higher separation score indicates that the embedding model has learned features that 
effectively distinguish between real and AI-generated images.
"""
        data = {
            'embedding_type': embedding_type,
            'separation_score': separation_score,
            'cluster_quality': cluster_quality
        }
        
        return self.add_section("Embedding Analysis", content, data)
        
    def add_conclusion(
        self,
        key_findings: List[str],
        recommendations: Optional[List[str]] = None
    ) -> 'ReportGenerator':
        """
        Add conclusion section.
        
        Args:
            key_findings: List of key findings
            recommendations: Optional list of recommendations
            
        Returns:
            self for method chaining
        """
        findings = "\n".join([f"- {f}" for f in key_findings])
        
        content = f"""
## Conclusions

### Key Findings

{findings}

"""
        if recommendations:
            recs = "\n".join([f"- {r}" for r in recommendations])
            content += f"""
### Recommendations

{recs}
"""
            
        data = {
            'key_findings': key_findings,
            'recommendations': recommendations
        }
        
        return self.add_section("Conclusions", content, data)
        
    def generate_markdown(self) -> str:
        """
        Generate the complete markdown report.
        
        Returns:
            Markdown-formatted report string
        """
        self.metadata['generated_at'] = datetime.now().isoformat()
        
        header = f"""# Explainable AI Report: {self.model_name}

**Generated**: {self.metadata['generated_at']}

---

"""
        content = header
        
        for section in self.sections:
            content += section['content'] + "\n\n---\n\n"
            
        return content
        
    def generate_json(self) -> Dict[str, Any]:
        """
        Generate JSON report with all data.
        
        Returns:
            Dictionary containing all report data
        """
        self.metadata['generated_at'] = datetime.now().isoformat()
        
        return {
            'metadata': self.metadata,
            'sections': [
                {
                    'title': s['title'],
                    'data': s['data']
                }
                for s in self.sections
            ]
        }
        
    def save(
        self,
        filepath: str,
        format: str = 'markdown'
    ) -> str:
        """
        Save the report to file.
        
        Args:
            filepath: Output file path
            format: Output format ('markdown' or 'json')
            
        Returns:
            The filepath where the report was saved
        """
        if format == 'markdown':
            content = self.generate_markdown()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format == 'json':
            content = self.generate_json()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return filepath
