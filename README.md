# DeepFake Detection with Explainable AI

A comprehensive toolkit for understanding and explaining decisions made by DeepFake detection models using embeddings from DINOv2 and OpenCLIP.

## Overview

This project provides explainability tools for linear probing classifiers trained on pre-extracted embeddings from:
- **DINOv2**: Self-supervised Vision Transformer features
- **OpenCLIP**: Contrastive Language-Image Pre-training features

The toolkit helps answer the question: *"What allows the model to distinguish between real and AI-generated images?"*

## Features

### 🔍 Feature Importance Analysis
- Coefficient-based importance ranking
- Directional analysis (which features indicate real vs fake)
- Per-sample contribution breakdown
- Distribution comparison between classes

### 📊 SHAP Analysis
- Game-theoretic feature importance
- Individual prediction explanations
- Global feature importance rankings
- Waterfall and beeswarm plots

### 🎨 Embedding Visualization
- PCA, t-SNE, and UMAP projections
- Decision boundary visualization
- Class separation analysis
- Embedding distance distributions

### 📝 Report Generation
- Automated markdown reports
- JSON export for further analysis
- Comprehensive model summaries

## Installation

```bash
# Clone the repository
git clone https://github.com/MomoWorlData/DeepFake-Detection-with-Explainable-IA.git
cd DeepFake-Detection-with-Explainable-IA

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.explainability import FeatureImportanceAnalyzer, SHAPExplainer, EmbeddingVisualizer
from src.utils import DataLoader

# Load your embeddings
data_loader = DataLoader()
data_loader.load_dinov2('path/to/dinov2_embeddings.npz')
data_loader.load_openclip('path/to/openclip_embeddings.npz')

# Or generate synthetic data for testing
data_loader.generate_synthetic_data(n_samples=10000)

# Feature Importance Analysis
analyzer = FeatureImportanceAnalyzer()
analyzer.fit_classifier(embeddings, labels)
analyzer.plot_feature_importance(top_k=20)

# SHAP Analysis
shap_explainer = SHAPExplainer(classifier=analyzer.classifier)
shap_explainer.fit(background_data=embeddings)
explanation = shap_explainer.explain_single(sample)

# Embedding Visualization
visualizer = EmbeddingVisualizer(embeddings, labels)
visualizer.fit_tsne()
visualizer.plot_2d(title="Embedding Space")
```

## Project Structure

```
DeepFake-Detection-with-Explainable-IA/
├── src/
│   ├── __init__.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── feature_importance.py   # Coefficient-based analysis
│   │   ├── shap_explainer.py       # SHAP-based explanations
│   │   └── embedding_visualizer.py # Visualization tools
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py          # Data loading utilities
│       └── report_generator.py     # Report generation
├── notebooks/
│   └── explainability_demo.ipynb   # Interactive demo
├── requirements.txt
└── README.md
```

## Usage Examples

### Analyzing a Single Prediction

```python
# Get detailed explanation for one image
analysis = analyzer.analyze_embedding(embedding, top_k=10)

print(f"Prediction: {analysis['prediction']}")
print(f"Confidence: {analysis['probability_fake']:.2%}")
print(f"Top contributing features: {analysis['feature_names'][:5]}")
```

### Comparing Real vs Fake Distributions

```python
# Visualize how features differ between classes
analyzer.compare_embeddings(
    real_embeddings=embeddings[labels == 0],
    fake_embeddings=embeddings[labels == 1],
    top_k=20
)
```

### SHAP Waterfall Plot

```python
# Show how each feature pushes prediction from base to final
shap_explainer.plot_waterfall(embedding, max_display=15)
```

## API Reference

### FeatureImportanceAnalyzer

| Method | Description |
|--------|-------------|
| `fit_classifier()` | Train a logistic regression on embeddings |
| `get_coefficient_importance()` | Get absolute coefficient values |
| `get_directional_importance()` | Get features by class direction |
| `analyze_embedding()` | Detailed analysis for single sample |
| `plot_feature_importance()` | Bar chart of top features |
| `plot_contribution_breakdown()` | Waterfall of feature contributions |

### SHAPExplainer

| Method | Description |
|--------|-------------|
| `fit()` | Initialize SHAP with background data |
| `explain()` | Calculate SHAP values |
| `explain_single()` | Detailed explanation for one sample |
| `plot_waterfall()` | SHAP waterfall plot |
| `plot_beeswarm()` | SHAP beeswarm plot |
| `get_global_importance()` | Mean absolute SHAP values |

### EmbeddingVisualizer

| Method | Description |
|--------|-------------|
| `fit_pca()` | PCA dimensionality reduction |
| `fit_tsne()` | t-SNE dimensionality reduction |
| `fit_umap()` | UMAP dimensionality reduction |
| `plot_2d()` | 2D scatter plot |
| `plot_decision_boundary()` | Visualize classifier boundary |
| `plot_embedding_distances()` | Distance distribution analysis |

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- torch >= 1.10.0
- shap >= 0.41.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- umap-learn >= 0.5.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised Vision Transformers
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open source CLIP implementation
- [SHAP](https://github.com/slundberg/shap) - SHapley Additive exPlanations