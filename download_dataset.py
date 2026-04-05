"""
PlantHealth AI - Dataset Download Script
==========================================
Downloads the New Plant Diseases Dataset from Kaggle using kagglehub
and organizes it in the project's data directory.

Usage:
    python download_dataset.py
    
Prerequisites:
    1. Run 'python setup_kaggle.py' first to configure Kaggle authentication
    2. Install kagglehub: pip install kagglehub
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub is not installed.")
    print("Install it with: pip install kagglehub")
    sys.exit(1)


# Configuration
DATASET_ID = "vipoooool/new-plant-diseases-dataset"
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXPECTED_FOLDERS = ["train", "valid", "test"]


def check_kaggle_auth() -> bool:
    """Check if Kaggle authentication is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if sys.platform == "win32":
        kaggle_json = Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("Error: Kaggle authentication not found.")
        print(f"Expected kaggle.json at: {kaggle_json}")
        print("\nPlease run 'python setup_kaggle.py' first.")
        return False
    
    return True


def download_dataset() -> Optional[Path]:
    """
    Download the dataset using kagglehub.
    
    Returns:
        Path to the downloaded dataset, or None if failed
    """
    print("=" * 60)
    print("PlantHealth AI - Dataset Download")
    print("=" * 60)
    print(f"\nDataset: {DATASET_ID}")
    print("Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print()
    
    try:
        print("Downloading dataset (this may take a while)...")
        print("File size: ~2.5 GB")
        print()
        
        # Download using kagglehub
        path = kagglehub.dataset_download(DATASET_ID)
        
        print(f"\nDownload complete!")
        print(f"Downloaded to: {path}")
        
        return Path(path)
    
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        return None


def find_dataset_root(download_path: Path) -> Optional[Path]:
    """
    Find the actual dataset root directory.
    The kagglehub download might have nested folders.
    
    Returns:
        Path to the directory containing train/valid/test folders
    """
    # Check if download_path directly contains the expected folders
    if all((download_path / folder).exists() for folder in ["train", "valid"]):
        return download_path
    
    # Search for the dataset folder
    for root, dirs, files in os.walk(download_path):
        root_path = Path(root)
        if "train" in dirs and "valid" in dirs:
            return root_path
    
    # Check for "New Plant Diseases Dataset" folder
    new_plant_dir = download_path / "New Plant Diseases Dataset"
    if new_plant_dir.exists():
        for root, dirs, files in os.walk(new_plant_dir):
            root_path = Path(root)
            if "train" in dirs and "valid" in dirs:
                return root_path
    
    return None


def copy_to_project(source_path: Path) -> bool:
    """
    Copy the dataset to the project's data directory.
    
    Args:
        source_path: Path to the downloaded dataset root
        
    Returns:
        True if successful, False otherwise
    """
    dest_path = DATA_DIR / "New Plant Diseases Dataset"
    
    # Check if already exists
    if dest_path.exists():
        print(f"\nDataset already exists at: {dest_path}")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping copy. Using existing dataset.")
            return True
        
        print("Removing existing dataset...")
        shutil.rmtree(dest_path)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying dataset to project directory...")
    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    print("\nThis may take a few minutes...")
    
    try:
        shutil.copytree(source_path, dest_path)
        print("Copy complete!")
        return True
    except Exception as e:
        print(f"Error copying dataset: {e}")
        return False


def verify_dataset(dataset_path: Path) -> bool:
    """
    Verify the dataset structure and print statistics.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if valid, False otherwise
    """
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    # Check required folders
    train_dir = dataset_path / "train"
    valid_dir = dataset_path / "valid"
    
    if not train_dir.exists():
        print(f"Error: Train directory not found: {train_dir}")
        return False
    
    if not valid_dir.exists():
        print(f"Error: Validation directory not found: {valid_dir}")
        return False
    
    # Count classes and images
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    valid_classes = sorted([d.name for d in valid_dir.iterdir() if d.is_dir()])
    
    print(f"\nTrain directory: {train_dir}")
    print(f"  Classes: {len(train_classes)}")
    
    train_total = 0
    for class_name in train_classes:
        class_dir = train_dir / class_name
        count = len(list(class_dir.glob("*")))
        train_total += count
    print(f"  Total images: {train_total:,}")
    
    print(f"\nValidation directory: {valid_dir}")
    print(f"  Classes: {len(valid_classes)}")
    
    valid_total = 0
    for class_name in valid_classes:
        class_dir = valid_dir / class_name
        count = len(list(class_dir.glob("*")))
        valid_total += count
    print(f"  Total images: {valid_total:,}")
    
    # Check test directory (optional)
    test_dir = dataset_path / "test"
    if test_dir.exists():
        test_count = len(list(test_dir.rglob("*.*")))
        print(f"\nTest directory: {test_dir}")
        print(f"  Total images: {test_count}")
    
    print("\n" + "=" * 60)
    print("Class Distribution")
    print("=" * 60)
    
    # Show class distribution
    print(f"\n{'Class Name':<55} {'Train':>8} {'Valid':>8}")
    print("-" * 73)
    
    for class_name in train_classes[:10]:  # Show first 10
        train_count = len(list((train_dir / class_name).glob("*")))
        valid_count = len(list((valid_dir / class_name).glob("*"))) if (valid_dir / class_name).exists() else 0
        print(f"{class_name:<55} {train_count:>8} {valid_count:>8}")
    
    if len(train_classes) > 10:
        print(f"... and {len(train_classes) - 10} more classes")
    
    print("\n" + "=" * 60)
    print("Dataset Ready!")
    print("=" * 60)
    print(f"\nTotal: {train_total + valid_total:,} images across {len(train_classes)} classes")
    print(f"\nNext steps:")
    print("1. Open 'notebooks/01_Data_Exploration.ipynb' to explore the data")
    print("2. Run training with 'notebooks/02_Custom_CNN.ipynb'")
    
    return True


def main():
    """Main function to orchestrate the download process."""
    
    # Check authentication
    if not check_kaggle_auth():
        sys.exit(1)
    
    # Check if dataset already exists
    existing_path = DATA_DIR / "New Plant Diseases Dataset"
    if existing_path.exists() and (existing_path / "train").exists():
        print(f"Dataset already exists at: {existing_path}")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            verify_dataset(existing_path)
            return
    
    # Download dataset
    download_path = download_dataset()
    if download_path is None:
        sys.exit(1)
    
    # Find dataset root
    dataset_root = find_dataset_root(download_path)
    if dataset_root is None:
        print("\nError: Could not find dataset structure (train/valid folders)")
        print(f"Downloaded to: {download_path}")
        print("Please manually check the downloaded files.")
        sys.exit(1)
    
    print(f"\nFound dataset at: {dataset_root}")
    
    # Copy to project directory
    if str(dataset_root) != str(existing_path):
        if not copy_to_project(dataset_root):
            sys.exit(1)
        verify_path = DATA_DIR / "New Plant Diseases Dataset"
    else:
        verify_path = dataset_root
    
    # Verify dataset
    if not verify_dataset(verify_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
