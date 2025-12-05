# DeepFake-Detection-with-Explainable-IA

This project provides tools for deepfake detection using explainable AI techniques with DINOv2 and OpenCLIP embeddings.

## Dataset Structure

The dataset should be organized as follows:
```
Dataset/
├── FAKE/     # 5k AI-generated fake images
└── REEL/     # 5k real images
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Interactive Shell Script (Linux/Mac)

```bash
chmod +x run.sh
./run.sh
```

### Option 2: Quick Start Script

Place your `Dataset.zip` file in the root directory, then run:

```bash
python quick_start.py
```

### Option 3: Manual Processing

Place your `Dataset.zip` file in the root directory, then run:

```bash
# Compute both DINOv2 and OpenCLIP embeddings
python process_dataset.py --compute-all

# Or compute them separately
python process_dataset.py --compute-dinov2
python process_dataset.py --compute-openclip
```

#### Options:
- `--dataset-zip PATH`: Path to Dataset.zip (default: `Dataset.zip`)
- `--extract-to PATH`: Directory to extract to (default: `./`)
- `--batch-size N`: Batch size for processing (default: 32)
- `--skip-extraction`: Skip extraction if dataset already extracted
- `--compute-dinov2`: Compute DINOv2 embeddings
- `--compute-openclip`: Compute OpenCLIP embeddings
- `--compute-all`: Compute both embeddings

### Option 4: Test and Analyze Embeddings

Open the Jupyter notebook to explore and test the embeddings:

```bash
jupyter notebook test_embeddings.ipynb
```

The notebook provides:
- Loading and inspecting embeddings
- Filtering REEL (real) images
- Visualizations (PCA, t-SNE)
- Statistics and similarity analysis
- Sample image display
- Exporting REEL-only embeddings

## Output Files

- `dinov2_embeddings.npz`: DINOv2 embeddings with labels and image paths
- `openclip_embeddings.npz`: OpenCLIP embeddings with labels and image paths
- `reel_only_dinov2_embeddings.npz`: DINOv2 embeddings for REEL images only
- `reel_only_openclip_embeddings.npz`: OpenCLIP embeddings for REEL images only

## Features

- **DINOv2 Embeddings**: Uses Facebook's DINOv2 vision transformer model
- **OpenCLIP Embeddings**: Uses OpenCLIP ViT-B-32 model pre-trained on LAION-2B
- **Batch Processing**: Efficient processing with configurable batch sizes
- **GPU Support**: Automatic GPU detection and usage when available
- **Comprehensive Testing**: Interactive notebook for embedding analysis