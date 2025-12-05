# Usage Examples

This document provides detailed examples for using the DeepFake Detection toolkit.

## Prerequisites

1. Place `Dataset.zip` in the project root directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Example 1: Quick Start (Recommended)

The easiest way to process the dataset and compute embeddings:

```bash
python quick_start.py
```

This will automatically:
- Check for Dataset.zip
- Verify dependencies
- Extract the dataset
- Compute both DINOv2 and OpenCLIP embeddings

## Example 2: Manual Processing with All Options

### Extract and compute both embeddings:
```bash
python process_dataset.py --compute-all --batch-size 32
```

### Compute only DINOv2 embeddings:
```bash
python process_dataset.py --compute-dinov2 --batch-size 32
```

### Compute only OpenCLIP embeddings:
```bash
python process_dataset.py --compute-openclip --batch-size 32
```

### Skip extraction if already done:
```bash
python process_dataset.py --skip-extraction --compute-all
```

### Custom dataset location:
```bash
python process_dataset.py \
  --dataset-zip /path/to/Dataset.zip \
  --extract-to /path/to/extract \
  --compute-all
```

## Example 3: Testing the Implementation

Run the test script to validate the processing logic:

```bash
python test_processing.py
```

This creates mock data and tests:
- Image collection from FAKE and REEL directories
- (Optional) Embedding computation with sample images

## Example 4: Working with Embeddings in Jupyter

After generating embeddings, explore them in the notebook:

```bash
jupyter notebook test_embeddings.ipynb
```

The notebook demonstrates:
- Loading embeddings from .npz files
- Filtering real (REEL) images
- PCA and t-SNE visualizations
- Statistical analysis
- Similarity computations
- Exporting REEL-only embeddings

## Example 5: Loading Embeddings in Python

```python
import numpy as np

# Load DINOv2 embeddings
dinov2_data = np.load('dinov2_embeddings.npz', allow_pickle=True)
embeddings = dinov2_data['embeddings']  # Shape: (N, 768)
labels = dinov2_data['labels']          # Shape: (N,) - 0=FAKE, 1=REEL
paths = dinov2_data['image_paths']      # Shape: (N,)

# Filter REEL images only
reel_mask = labels == 1
reel_embeddings = embeddings[reel_mask]
reel_paths = paths[reel_mask]

print(f"Total images: {len(embeddings)}")
print(f"REEL images: {len(reel_embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

## Example 6: Using Embeddings for Classification

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load embeddings
data = np.load('dinov2_embeddings.npz', allow_pickle=True)
X = data['embeddings']
y = data['labels']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['FAKE', 'REEL']))
```

## Example 7: Batch Processing with Custom Settings

For large datasets or limited GPU memory:

```bash
# Use smaller batch size for limited memory
python process_dataset.py --compute-all --batch-size 8

# Or process sequentially to avoid memory issues
python process_dataset.py --compute-dinov2 --batch-size 16
python process_dataset.py --skip-extraction --compute-openclip --batch-size 16
```

## Output Files

After successful processing, you'll have:

- `dinov2_embeddings.npz` - DINOv2 embeddings (768 dimensions)
- `openclip_embeddings.npz` - OpenCLIP embeddings (512 dimensions)
- `reel_only_dinov2_embeddings.npz` - REEL images only (created by notebook)
- `reel_only_openclip_embeddings.npz` - REEL images only (created by notebook)

Each .npz file contains:
- `embeddings`: NumPy array of shape (N, D) where D is embedding dimension
- `labels`: NumPy array of shape (N,) with 0=FAKE, 1=REEL
- `image_paths`: NumPy array of shape (N,) with original image paths

## Troubleshooting

### Out of Memory Errors
Reduce batch size:
```bash
python process_dataset.py --compute-all --batch-size 4
```

### Model Download Issues
Models are downloaded automatically on first use. Ensure you have:
- Stable internet connection
- Sufficient disk space (~2GB for models)

### CUDA Out of Memory
The script automatically falls back to CPU if CUDA runs out of memory.
For faster CPU processing, use smaller batch sizes.

### Dataset Not Found
Ensure Dataset.zip is in the current directory or specify path:
```bash
python process_dataset.py --dataset-zip /path/to/Dataset.zip --compute-all
```