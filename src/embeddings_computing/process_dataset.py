"""
Script to process the Dataset.zip file:
1. Extract images from FAKE and REEL subdirectories
2. Compute DINOv2 embeddings and save to dinov2_embeddings.npz
3. Compute OpenCLIP embeddings and save to openclip_embeddings.npz
"""

import os
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


def extract_dataset(zip_path, extract_to="./"):
    """
    Extract Dataset.zip to the specified directory.
    
    Args:
        zip_path: Path to Dataset.zip
        extract_to: Directory to extract to
    
    Returns:
        Path to extracted dataset directory
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")
    return os.path.join(extract_to, "Dataset")


def collect_images(dataset_path):
    """
    Collect all images from FAKE and REEL subdirectories.
    
    Args:
        dataset_path: Path to Dataset directory
    
    Returns:
        List of tuples (image_path, label) where label is 0 for FAKE and 1 for REEL
    """
    image_data = []
    
    # Collect FAKE images (label 0)
    fake_dir = os.path.join(dataset_path, "FAKE")
    if os.path.exists(fake_dir):
        print(f"Collecting FAKE images from {fake_dir}...")
        for img_file in os.listdir(fake_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_data.append((os.path.join(fake_dir, img_file), 0))
    
    # Collect REEL images (label 1)
    reel_dir = os.path.join(dataset_path, "REEL")
    if os.path.exists(reel_dir):
        print(f"Collecting REEL images from {reel_dir}...")
        for img_file in os.listdir(reel_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_data.append((os.path.join(reel_dir, img_file), 1))
    
    print(f"Total images collected: {len(image_data)}")
    print(f"  FAKE: {sum(1 for _, label in image_data if label == 0)}")
    print(f"  REEL: {sum(1 for _, label in image_data if label == 1)}")
    
    return image_data


def compute_dinov2_embeddings(image_data, output_path="dinov2_embeddings.npz", batch_size=32):
    """
    Compute DINOv2 embeddings for all images.
    
    Args:
        image_data: List of tuples (image_path, label)
        output_path: Path to save embeddings
        batch_size: Batch size for processing
    """
    print("\n=== Computing DINOv2 Embeddings ===")
    
    # Import torch here to avoid dependency issues
    import torch
    
    # Load DINOv2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        from transformers import AutoImageProcessor, AutoModel
        
        model_name = "facebook/dinov2-base"
        print(f"Loading {model_name}...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        
        embeddings_list = []
        labels_list = []
        image_paths_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_data), batch_size), desc="Processing batches"):
                batch_data = image_data[i:i+batch_size]
                batch_images = []
                batch_labels = []
                batch_paths = []
                
                for img_path, label in batch_data:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        batch_images.append(img)
                        batch_labels.append(label)
                        batch_paths.append(img_path)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
                
                if batch_images:
                    # Process images
                    inputs = processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Get embeddings
                    outputs = model(**inputs)
                    # Use CLS token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                    
                    embeddings_list.append(batch_embeddings)
                    labels_list.extend(batch_labels)
                    image_paths_list.extend(batch_paths)
        
        # Check if we have any embeddings
        if not embeddings_list:
            raise ValueError(
                "No images were successfully processed. Please check your dataset.\n"
                "Possible causes:\n"
                "  - Images may be corrupted\n"
                "  - Unsupported image formats (supported: PNG, JPG, JPEG, BMP, GIF)\n"
                "  - Incorrect dataset path\n"
                "  - Empty FAKE or REEL directories"
            )
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list)
        labels = np.array(labels_list)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Save embeddings
        np.savez(output_path, 
                 embeddings=embeddings, 
                 labels=labels,
                 image_paths=np.array(image_paths_list))
        print(f"DINOv2 embeddings saved to {output_path}")
        
    except Exception as e:
        print(f"Error computing DINOv2 embeddings: {e}")
        raise


def compute_openclip_embeddings(image_data, output_path="openclip_embeddings.npz", batch_size=32):
    """
    Compute OpenCLIP embeddings for all images.
    
    Args:
        image_data: List of tuples (image_path, label)
        output_path: Path to save embeddings
        batch_size: Batch size for processing
    """
    print("\n=== Computing OpenCLIP Embeddings ===")
    
    # Import torch here to avoid dependency issues
    import torch
    
    # Load OpenCLIP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        import open_clip
        
        model_name = "ViT-B-32"
        pretrained = "laion2b_s34b_b79k"
        print(f"Loading {model_name} with {pretrained}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        
        embeddings_list = []
        labels_list = []
        image_paths_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_data), batch_size), desc="Processing batches"):
                batch_data = image_data[i:i+batch_size]
                batch_images = []
                batch_labels = []
                batch_paths = []
                
                for img_path, label in batch_data:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        batch_images.append(preprocess(img))
                        batch_labels.append(label)
                        batch_paths.append(img_path)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
                
                if batch_images:
                    # Stack images into batch
                    batch_tensor = torch.stack(batch_images).to(device)
                    
                    # Get embeddings
                    batch_embeddings = model.encode_image(batch_tensor).cpu().numpy()
                    
                    embeddings_list.append(batch_embeddings)
                    labels_list.extend(batch_labels)
                    image_paths_list.extend(batch_paths)
        
        # Check if we have any embeddings
        if not embeddings_list:
            raise ValueError(
                "No images were successfully processed. Please check your dataset.\n"
                "Possible causes:\n"
                "  - Images may be corrupted\n"
                "  - Unsupported image formats (supported: PNG, JPG, JPEG, BMP, GIF)\n"
                "  - Incorrect dataset path\n"
                "  - Empty FAKE or REEL directories"
            )
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list)
        labels = np.array(labels_list)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Save embeddings
        np.savez(output_path, 
                 embeddings=embeddings, 
                 labels=labels,
                 image_paths=np.array(image_paths_list))
        print(f"OpenCLIP embeddings saved to {output_path}")
        
    except Exception as e:
        print(f"Error computing OpenCLIP embeddings: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Process Dataset.zip and compute embeddings')
    parser.add_argument('--dataset-zip', type=str, default='Dataset.zip',
                        help='Path to Dataset.zip file')
    parser.add_argument('--extract-to', type=str, default='./',
                        help='Directory to extract dataset to')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip extraction if dataset already extracted')
    parser.add_argument('--compute-dinov2', action='store_true', default=False,
                        help='Compute DINOv2 embeddings')
    parser.add_argument('--compute-openclip', action='store_true', default=False,
                        help='Compute OpenCLIP embeddings')
    parser.add_argument('--compute-all', action='store_true', default=False,
                        help='Compute both DINOv2 and OpenCLIP embeddings')
    
    args = parser.parse_args()
    
    # If compute-all is set, enable both
    if args.compute_all:
        args.compute_dinov2 = True
        args.compute_openclip = True
    
    # Extract dataset if needed
    if not args.skip_extraction:
        if not os.path.exists(args.dataset_zip):
            print(f"Error: {args.dataset_zip} not found!")
            print("Please place Dataset.zip in the current directory or specify --dataset-zip path")
            return
        dataset_path = extract_dataset(args.dataset_zip, args.extract_to)
    else:
        dataset_path = os.path.join(args.extract_to, "Dataset")
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset directory not found at {dataset_path}")
            return
    
    # Collect images
    image_data = collect_images(dataset_path)
    
    if len(image_data) == 0:
        print("No images found in the dataset!")
        return
    
    # Compute embeddings based on flags
    if args.compute_dinov2:
        compute_dinov2_embeddings(image_data, batch_size=args.batch_size)
    
    if args.compute_openclip:
        compute_openclip_embeddings(image_data, batch_size=args.batch_size)
    
    if not args.compute_dinov2 and not args.compute_openclip:
        print("\nNote: No embeddings computed. Use --compute-all, --compute-dinov2, or --compute-openclip")
    
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()