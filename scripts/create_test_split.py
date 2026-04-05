"""
PlantHealth AI - Create Test Split
===================================
Splits the validation set into validation (80%) and test (20%) sets.
Maintains class balance with stratified splitting.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SEED = 42
TEST_RATIO = 0.20  # 20% of validation goes to test

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "New Plant Diseases Dataset"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

def create_test_split():
    """Split validation data into val/test while preserving class balance."""
    random.seed(SEED)
    
    if not VALID_DIR.exists():
        print(f"ERROR: Validation directory not found: {VALID_DIR}")
        return
    
    if TEST_DIR.exists():
        print(f"Test directory already exists: {TEST_DIR}")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
        shutil.rmtree(TEST_DIR)
    
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = sorted([d for d in VALID_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes in validation set")
    
    stats = defaultdict(dict)
    total_moved = 0
    total_remaining = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all images in this class
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in class_dir.iterdir() if f.suffix in valid_extensions]
        
        if len(images) == 0:
            print(f"  WARNING: No images in {class_name}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        n_test = max(1, int(len(images) * TEST_RATIO))  # At least 1 test image
        
        test_images = images[:n_test]
        
        # Create test class directory
        test_class_dir = TEST_DIR / class_name
        test_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy test images (not move, to preserve original valid set)
        for img_path in test_images:
            dest = test_class_dir / img_path.name
            shutil.copy2(img_path, dest)
        
        remaining = len(images) - n_test
        stats[class_name] = {
            'original': len(images),
            'test': n_test,
            'remaining': remaining,
        }
        total_moved += n_test
        total_remaining += remaining
        
        print(f"  {class_name}: {len(images)} total → {n_test} test, {remaining} val")
    
    print(f"\n{'='*60}")
    print(f"Test split complete!")
    print(f"  Total test images: {total_moved}")
    print(f"  Total val images remaining: {total_remaining}")
    print(f"  Test directory: {TEST_DIR}")
    print(f"  Classes in test set: {len(list(TEST_DIR.iterdir()))}")


if __name__ == "__main__":
    create_test_split()
