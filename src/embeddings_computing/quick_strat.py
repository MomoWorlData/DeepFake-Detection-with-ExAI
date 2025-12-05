#!/usr/bin/env python3
"""
Quick start script for dataset processing and embedding generation.
This script provides a simple interface to the main functionality.
"""

import os
import sys


def print_usage():
    """Print usage instructions."""
    print("=" * 70)
    print("DeepFake Detection - Dataset Processing & Embedding Generation")
    print("=" * 70)
    print()
    print("SETUP:")
    print("  1. Ensure Dataset.zip is in the current directory")
    print("  2. Install dependencies: pip install -r requirements.txt")
    print()
    print("USAGE:")
    print("  python quick_start.py")
    print()
    print("This will:")
    print("  - Extract Dataset.zip")
    print("  - Collect images from FAKE and REEL directories")
    print("  - Compute DINOv2 embeddings (saved to dinov2_embeddings.npz)")
    print("  - Compute OpenCLIP embeddings (saved to openclip_embeddings.npz)")
    print()
    print("TESTING:")
    print("  jupyter notebook test_embeddings.ipynb")
    print()
    print("=" * 70)


def check_dataset():
    """Check if Dataset.zip exists."""
    if not os.path.exists('Dataset.zip'):
        print("ERROR: Dataset.zip not found in current directory!")
        print()
        print("Please place Dataset.zip in the current directory and try again.")
        return False
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required = ['torch', 'transformers', 'open_clip_torch', 'numpy', 'tqdm', 'PIL']
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'open_clip_torch':
                __import__('open_clip')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("Install dependencies with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main execution function."""
    print_usage()
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✓ All dependencies installed")
    print()
    
    # Import and run main processing
    print("Starting dataset processing...")
    print("This may take a while depending on your hardware...")
    print()
    
    # Run the main script
    import subprocess
    result = subprocess.run([
        sys.executable, 
        'process_dataset.py',
        '--compute-all'
    ])
    
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("SUCCESS! Embeddings generated successfully.")
        print()
        print("Output files:")
        print("  - dinov2_embeddings.npz")
        print("  - openclip_embeddings.npz")
        print()
        print("Next steps:")
        print("  jupyter notebook test_embeddings.ipynb")
        print("=" * 70)
    else:
        print()
        print("ERROR: Processing failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()