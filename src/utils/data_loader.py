"""
Data loading utilities for embeddings and labels.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split as sklearn_split


def load_embeddings(
    filepath: str,
    format: str = 'auto'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load embeddings from file.
    
    Supports .npy, .npz, and .pt (PyTorch) formats.
    
    Args:
        filepath: Path to the embeddings file
        format: File format ('auto', 'npy', 'npz', 'pt')
        
    Returns:
        Tuple of (embeddings, labels) where labels may be None
    """
    filepath = Path(filepath)
    
    if format == 'auto':
        format = filepath.suffix[1:]  # Remove the dot
        
    if format == 'npy':
        embeddings = np.load(filepath)
        return embeddings, None
        
    elif format == 'npz':
        data = np.load(filepath)
        embeddings = data.get('embeddings', data.get('arr_0'))
        labels = data.get('labels', data.get('arr_1', None))
        return embeddings, labels
        
    elif format == 'pt':
        import torch
        data = torch.load(filepath, map_location='cpu')
        if isinstance(data, dict):
            embeddings = data.get('embeddings', data.get('features'))
            labels = data.get('labels', None)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
        else:
            embeddings = data.numpy() if isinstance(data, torch.Tensor) else data
            labels = None
        return embeddings, labels
        
    else:
        raise ValueError(f"Unsupported format: {format}")


class DataLoader:
    """
    Utility class for loading and managing embeddings datasets.
    
    Attributes:
        dinov2_embeddings: DINOv2 embeddings
        openclip_embeddings: OpenCLIP embeddings
        labels: Class labels (0=real, 1=fake)
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.dinov2_embeddings = None
        self.openclip_embeddings = None
        self.labels = None
        self.metadata = {}
        
    def load_dinov2(self, filepath: str) -> 'DataLoader':
        """
        Load DINOv2 embeddings.
        
        Args:
            filepath: Path to DINOv2 embeddings file
            
        Returns:
            self for method chaining
        """
        self.dinov2_embeddings, labels = load_embeddings(filepath)
        if labels is not None:
            self.labels = labels
        self.metadata['dinov2_path'] = filepath
        self.metadata['dinov2_dim'] = self.dinov2_embeddings.shape[1]
        return self
        
    def load_openclip(self, filepath: str) -> 'DataLoader':
        """
        Load OpenCLIP embeddings.
        
        Args:
            filepath: Path to OpenCLIP embeddings file
            
        Returns:
            self for method chaining
        """
        self.openclip_embeddings, labels = load_embeddings(filepath)
        if labels is not None:
            self.labels = labels
        self.metadata['openclip_path'] = filepath
        self.metadata['openclip_dim'] = self.openclip_embeddings.shape[1]
        return self
        
    def load_labels(self, filepath: str, column: Optional[str] = None) -> 'DataLoader':
        """
        Load labels from a separate file.
        
        Args:
            filepath: Path to labels file
            column: Column name or index for CSV files. If None, uses last column.
            
        Returns:
            self for method chaining
        """
        filepath = Path(filepath)
        if filepath.suffix == '.npy':
            self.labels = np.load(filepath)
        elif filepath.suffix == '.txt':
            self.labels = np.loadtxt(filepath, dtype=int)
        elif filepath.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            if column is not None:
                self.labels = df[column].values
            else:
                self.labels = df.iloc[:, -1].values
        else:
            raise ValueError(f"Unsupported label format: {filepath.suffix}")
        return self
        
    def generate_synthetic_data(
        self,
        n_samples: int = 10000,
        dinov2_dim: int = 768,
        openclip_dim: int = 512,
        random_state: int = 42
    ) -> 'DataLoader':
        """
        Generate synthetic embeddings for testing purposes.
        
        Creates embeddings where real and fake classes have slightly
        different distributions to simulate a realistic scenario.
        
        Args:
            n_samples: Total number of samples (half real, half fake)
            dinov2_dim: Dimension of DINOv2 embeddings
            openclip_dim: Dimension of OpenCLIP embeddings
            random_state: Random seed for reproducibility
            
        Returns:
            self for method chaining
        """
        np.random.seed(random_state)
        
        n_real = n_samples // 2
        n_fake = n_samples - n_real
        
        # Create labels
        self.labels = np.array([0] * n_real + [1] * n_fake)
        
        # Generate DINOv2 embeddings with class-specific patterns
        # Real images: centered around one mean
        # Fake images: slightly shifted mean with different variance
        real_dinov2 = np.random.randn(n_real, dinov2_dim) * 0.5
        fake_dinov2 = np.random.randn(n_fake, dinov2_dim) * 0.6 + 0.1
        
        # Add some discriminative dimensions
        discriminative_dims = np.random.choice(dinov2_dim, size=50, replace=False)
        real_dinov2[:, discriminative_dims] += np.random.randn(n_real, 50) * 0.3 - 0.5
        fake_dinov2[:, discriminative_dims] += np.random.randn(n_fake, 50) * 0.3 + 0.5
        
        self.dinov2_embeddings = np.vstack([real_dinov2, fake_dinov2])
        
        # Generate OpenCLIP embeddings similarly
        real_openclip = np.random.randn(n_real, openclip_dim) * 0.5
        fake_openclip = np.random.randn(n_fake, openclip_dim) * 0.55 + 0.15
        
        discriminative_dims_clip = np.random.choice(openclip_dim, size=40, replace=False)
        real_openclip[:, discriminative_dims_clip] += np.random.randn(n_real, 40) * 0.25 - 0.4
        fake_openclip[:, discriminative_dims_clip] += np.random.randn(n_fake, 40) * 0.25 + 0.4
        
        self.openclip_embeddings = np.vstack([real_openclip, fake_openclip])
        
        # Shuffle the data
        shuffle_idx = np.random.permutation(n_samples)
        self.dinov2_embeddings = self.dinov2_embeddings[shuffle_idx]
        self.openclip_embeddings = self.openclip_embeddings[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        
        self.metadata['synthetic'] = True
        self.metadata['n_samples'] = n_samples
        self.metadata['dinov2_dim'] = dinov2_dim
        self.metadata['openclip_dim'] = openclip_dim
        
        return self
        
    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed
            
        Returns:
            Dictionary with train/test splits for each embedding type
        """
        result = {}
        
        if self.dinov2_embeddings is not None:
            X_train, X_test, y_train, y_test = sklearn_split(
                self.dinov2_embeddings, self.labels,
                test_size=test_size, random_state=random_state,
                stratify=self.labels
            )
            result['dinov2_train'] = X_train
            result['dinov2_test'] = X_test
            result['labels_train'] = y_train
            result['labels_test'] = y_test
            
        if self.openclip_embeddings is not None:
            X_train, X_test, _, _ = sklearn_split(
                self.openclip_embeddings, self.labels,
                test_size=test_size, random_state=random_state,
                stratify=self.labels
            )
            result['openclip_train'] = X_train
            result['openclip_test'] = X_test
            
        return result
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        summary = {
            'metadata': self.metadata
        }
        
        if self.dinov2_embeddings is not None:
            summary['dinov2'] = {
                'shape': self.dinov2_embeddings.shape,
                'mean': float(self.dinov2_embeddings.mean()),
                'std': float(self.dinov2_embeddings.std()),
                'min': float(self.dinov2_embeddings.min()),
                'max': float(self.dinov2_embeddings.max())
            }
            
        if self.openclip_embeddings is not None:
            summary['openclip'] = {
                'shape': self.openclip_embeddings.shape,
                'mean': float(self.openclip_embeddings.mean()),
                'std': float(self.openclip_embeddings.std()),
                'min': float(self.openclip_embeddings.min()),
                'max': float(self.openclip_embeddings.max())
            }
            
        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            summary['labels'] = {
                'total': len(self.labels),
                'distribution': dict(zip(['real' if u == 0 else 'fake' for u in unique], 
                                        counts.tolist()))
            }
            
        return summary
