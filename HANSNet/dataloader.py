"""
dataloader.py - Dataset Metadata & Slice Processing for HANS-Net

This module provides utilities for parsing the liver CT dataset structure:
    Dataset/
    ├── train_images/
    │   └── train_images/
    │       └── volume-<case>_slice_<number>.jpg
    └── train_masks/
        └── volume-<case>_slice_<number>.jpg
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF


# =============================================================================
# Helper Functions
# =============================================================================

def get_case_id_from_filename(fname: str) -> str:
    """
    Extract case ID from filename.
    
    Given: 'volume-13_slice_55.jpg'
    Return: 'volume-13'
    
    Args:
        fname: Filename (with or without path, with extension)
        
    Returns:
        Case ID string (e.g., 'volume-13')
        
    Raises:
        ValueError: If filename format doesn't match expected pattern
    """
    # Get basename in case full path is provided
    basename = os.path.basename(fname)
    
    # Strip file extension
    name_no_ext = os.path.splitext(basename)[0]
    
    # Split based on '_slice_'
    if '_slice_' not in name_no_ext:
        raise ValueError(
            f"Invalid filename format: '{fname}'. "
            f"Expected format: 'volume-<case>_slice_<number>.jpg'"
        )
    
    parts = name_no_ext.split('_slice_')
    
    if len(parts) != 2:
        raise ValueError(
            f"Invalid filename format: '{fname}'. "
            f"Multiple '_slice_' occurrences found."
        )
    
    case_id = parts[0]
    
    # Validate case_id format (should be 'volume-<number>')
    if not re.match(r'^volume-\d+$', case_id):
        raise ValueError(
            f"Invalid case ID format: '{case_id}'. "
            f"Expected format: 'volume-<number>'"
        )
    
    return case_id


def get_slice_index(fname: str) -> int:
    """
    Extract slice index from filename.
    
    Given: 'volume-13_slice_55.jpg'
    Return: 55 (int)
    
    Args:
        fname: Filename (with or without path, with extension)
        
    Returns:
        Slice index as integer
        
    Raises:
        ValueError: If parsing fails
    """
    # Get basename in case full path is provided
    basename = os.path.basename(fname)
    
    # Strip file extension
    name_no_ext = os.path.splitext(basename)[0]
    
    # Split based on '_slice_'
    if '_slice_' not in name_no_ext:
        raise ValueError(
            f"Invalid filename format: '{fname}'. "
            f"Expected format: 'volume-<case>_slice_<number>.jpg'"
        )
    
    parts = name_no_ext.split('_slice_')
    
    if len(parts) != 2:
        raise ValueError(
            f"Invalid filename format: '{fname}'. "
            f"Multiple '_slice_' occurrences found."
        )
    
    slice_str = parts[1]
    
    # Parse the number after '_slice_'
    try:
        slice_idx = int(slice_str)
    except ValueError:
        raise ValueError(
            f"Invalid slice index: '{slice_str}' in filename '{fname}'. "
            f"Expected an integer after '_slice_'."
        )
    
    return slice_idx


# =============================================================================
# Data Class
# =============================================================================

@dataclass
class SliceMeta:
    """Metadata for a single slice (image + mask pair)."""
    img_path: str       # Full absolute path to image file
    mask_path: str      # Full absolute path to mask file
    case_id: str        # Case identifier (e.g., 'volume-13')
    slice_idx: int      # Slice index within the volume


# =============================================================================
# Metadata Construction
# =============================================================================

def build_slice_metadata(img_root: str, mask_root: str) -> List[SliceMeta]:
    """
    Scan folders, pair matching image/mask filenames, and extract metadata.
    
    Args:
        img_root: Path to folder containing image files
        mask_root: Path to folder containing mask files
        
    Returns:
        List of SliceMeta objects (unsorted)
        
    Raises:
        FileNotFoundError: If img_root or mask_root doesn't exist
        ValueError: If an image has no corresponding mask
    """
    # Validate paths exist
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Image root directory not found: '{img_root}'")
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Mask root directory not found: '{mask_root}'")
    
    # Get absolute paths
    img_root = os.path.abspath(img_root)
    mask_root = os.path.abspath(mask_root)
    
    # List all image files
    img_files = [
        f for f in os.listdir(img_root)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    # Build set of mask filenames for quick lookup
    mask_files_set = set(
        f for f in os.listdir(mask_root)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    
    metadata_list: List[SliceMeta] = []
    
    for img_fname in img_files:
        # Check that corresponding mask exists
        if img_fname not in mask_files_set:
            raise ValueError(
                f"No corresponding mask found for image: '{img_fname}'"
            )
        
        # Build full absolute paths
        img_path = os.path.join(img_root, img_fname)
        mask_path = os.path.join(mask_root, img_fname)
        
        # Extract metadata using helper functions
        case_id = get_case_id_from_filename(img_fname)
        slice_idx = get_slice_index(img_fname)
        
        # Create SliceMeta object
        meta = SliceMeta(
            img_path=img_path,
            mask_path=mask_path,
            case_id=case_id,
            slice_idx=slice_idx
        )
        
        metadata_list.append(meta)
    
    return metadata_list


# =============================================================================
# PyTorch Dataset
# =============================================================================

class LITSSliceDataset(Dataset):
    """
    PyTorch Dataset for LITS liver CT slices.
    
    Returns triplets of consecutive slices (t-1, t, t+1) with edge replication,
    along with the center slice's segmentation mask.
    
    Output shapes:
        imgs_3:      [3, 1, H, W]  - three consecutive grayscale slices
        mask_center: [1, H, W]     - binary mask for center slice
    """
    
    def __init__(
        self,
        img_root: str,
        mask_root: str,
        img_size: Tuple[int, int] = (128, 128),
    ):
        """
        Initialize the dataset.
        
        Args:
            img_root: Path to folder containing image files
            mask_root: Path to folder containing mask files
            img_size: Target size (H, W) for resizing images and masks
        """
        self.img_size = img_size
        
        # Build metadata from folder structure
        metadata_list = build_slice_metadata(img_root, mask_root)
        
        # Group slices by case_id
        case_to_slices: Dict[str, List[SliceMeta]] = {}
        for meta in metadata_list:
            if meta.case_id not in case_to_slices:
                case_to_slices[meta.case_id] = []
            case_to_slices[meta.case_id].append(meta)
        
        # Sort each case's slices by slice_idx (numeric, not filename)
        for cid in case_to_slices:
            case_to_slices[cid].sort(key=lambda m: m.slice_idx)
        
        self.case_to_slices = case_to_slices
        
        # Build samples list: (case_id, center_idx) for every possible center
        self.samples: List[Tuple[str, int]] = []
        for case_id, slices in self.case_to_slices.items():
            num_slices = len(slices)
            for center_idx in range(num_slices):
                self.samples.append((case_id, center_idx))
    
    def __len__(self) -> int:
        """Return total number of samples (all slices across all cases)."""
        return len(self.samples)
    
    def _get_triplet_indices(
        self, num_slices: int, center_idx: int
    ) -> Tuple[int, int, int]:
        """
        Get indices for triplet (t-1, t, t+1) with edge replication.
        
        Args:
            num_slices: Total number of slices in the case
            center_idx: Index of the center slice
            
        Returns:
            Tuple of (prev_idx, center_idx, next_idx)
            
        Examples:
            center_idx == 0     -> (0, 0, 1)
            center_idx == N-1   -> (N-2, N-1, N-1)
        """
        prev_idx = max(0, center_idx - 1)
        next_idx = min(num_slices - 1, center_idx + 1)
        return (prev_idx, center_idx, next_idx)
    
    def _load_slice(self, path: str) -> torch.Tensor:
        """
        Load a single image slice.
        
        Args:
            path: Full path to the image file
            
        Returns:
            Tensor of shape [1, H, W] with values in [0, 1]
        """
        # Open image and convert to grayscale
        img = Image.open(path).convert('L')
        
        # Resize to target size (W, H) - PIL uses (width, height)
        img = img.resize(
            (self.img_size[1], self.img_size[0]),
            resample=Image.BILINEAR
        )
        
        # Convert to tensor [1, H, W] in [0, 1]
        img_tensor = TF.to_tensor(img)
        
        return img_tensor
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """
        Load a single mask slice.
        
        Args:
            path: Full path to the mask file
            
        Returns:
            Binary tensor of shape [1, H, W] with values {0, 1}
        """
        # Open mask and convert to grayscale
        mask = Image.open(path).convert('L')
        
        # Resize with NEAREST interpolation to preserve labels
        mask = mask.resize(
            (self.img_size[1], self.img_size[0]),
            resample=Image.NEAREST
        )
        
        # Convert to tensor [1, H, W] in [0, 1]
        mask_tensor = TF.to_tensor(mask)
        
        # Binarize: values > 0.5 become 1, else 0
        mask_tensor = (mask_tensor > 0.5).float()
        
        return mask_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample: triplet of slices + center mask.
        
        Args:
            idx: Sample index
            
        Returns:
            imgs_3: Tensor of shape [3, 1, H, W] (slices t-1, t, t+1)
            mask_center: Tensor of shape [1, H, W] (mask of center slice)
        """
        # Get case_id and center_idx for this sample
        case_id, center_idx = self.samples[idx]
        
        # Get sorted slice list for this case
        slices = self.case_to_slices[case_id]
        num_slices = len(slices)
        
        # Get triplet indices with edge replication
        i_prev, i_center, i_next = self._get_triplet_indices(num_slices, center_idx)
        
        # Load three image slices
        img_prev = self._load_slice(slices[i_prev].img_path)      # [1, H, W]
        img_center = self._load_slice(slices[i_center].img_path)  # [1, H, W]
        img_next = self._load_slice(slices[i_next].img_path)      # [1, H, W]
        
        # Stack along new dimension: [3, 1, H, W]
        imgs_3 = torch.stack([img_prev, img_center, img_next], dim=0)
        
        # Load center mask: [1, H, W]
        mask_center = self._load_mask(slices[i_center].mask_path)
        
        return imgs_3, mask_center


# =============================================================================
# DataLoader Builder
# =============================================================================

def build_dataloaders(
    dataset_root: str,
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 4,
    train_ratio: float = 0.8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from the dataset.
    
    Args:
        dataset_root: Root path to the dataset folder containing
                      'train_images/train_images/' and 'train_masks/'
        img_size: Target size (H, W) for resizing images and masks
        batch_size: Batch size for both loaders
        train_ratio: Fraction of data to use for training (default 0.8)
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Construct paths based on folder structure
    img_root = os.path.join(dataset_root, "train_images", "train_images")
    mask_root = os.path.join(dataset_root, "train_masks")
    
    # Instantiate full dataset
    full_dataset = LITSSliceDataset(
        img_root=img_root,
        mask_root=mask_root,
        img_size=img_size,
    )
    
    # Compute train/val split sizes
    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train
    
    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# =============================================================================
# Dataset Structure Demo / Validation
# =============================================================================

def demo_dataset_structure(dataset_root: str) -> None:
    """
    Perform a safe, dry-run validation of the dataset structure.
    
    This function checks whether the expected folders exist and reports
    metadata statistics without loading actual pixel data.
    
    Args:
        dataset_root: Root path to the dataset folder
        
    Returns:
        None - prints diagnostic information
        
    Note:
        This function will NOT throw errors if the dataset is missing.
        It gracefully reports whether folders exist and how many slices
        are detected.
    """
    # Construct expected paths
    img_root = dataset_root + "/train_images/train_images/"
    mask_root = dataset_root + "/train_masks/"
    
    print("=" * 60)
    print("Dataset Structure Validation")
    print("=" * 60)
    print(f"\nDataset root: {dataset_root}")
    print(f"Image folder: {img_root}")
    print(f"Mask folder:  {mask_root}")
    print()
    
    # Check if directories exist
    img_exists = os.path.isdir(img_root)
    mask_exists = os.path.isdir(mask_root)
    
    print("-" * 40)
    print("Folder Status:")
    print("-" * 40)
    
    if img_exists:
        print(f"  [✓] Image folder exists")
    else:
        print(f"  [✗] Image folder NOT found — dataset may not yet be downloaded.")
    
    if mask_exists:
        print(f"  [✓] Mask folder exists")
    else:
        print(f"  [✗] Mask folder NOT found — dataset may not yet be downloaded.")
    
    # Return early if either folder is missing
    if not img_exists or not mask_exists:
        print("\n⚠ Cannot proceed with metadata extraction — folders missing.")
        print("=" * 60)
        return
    
    # Folders exist — build metadata
    print("\n" + "-" * 40)
    print("Metadata Extraction:")
    print("-" * 40)
    
    metadata_list = build_slice_metadata(img_root, mask_root)
    
    # Count total slices
    total_slices = len(metadata_list)
    print(f"\nTotal slices detected: {total_slices}")
    
    # Group by case_id to count unique cases
    case_to_slices: dict = {}
    for meta in metadata_list:
        if meta.case_id not in case_to_slices:
            case_to_slices[meta.case_id] = []
        case_to_slices[meta.case_id].append(meta.slice_idx)
    
    num_cases = len(case_to_slices)
    print(f"Number of unique cases: {num_cases}")
    
    # Show details for up to 2 cases
    print("\n" + "-" * 40)
    print("Sample Cases (up to 2):")
    print("-" * 40)
    
    case_ids = sorted(case_to_slices.keys())[:2]
    for case_id in case_ids:
        slice_indices = sorted(case_to_slices[case_id])
        first_six = slice_indices[:6]
        total_in_case = len(slice_indices)
        
        print(f"\n  Case: {case_id}")
        print(f"    Total slices in case: {total_in_case}")
        print(f"    Slices (first 6): {first_six}")
    
    print("\n" + "=" * 60)
    print("Validation complete — no pixel data loaded.")
    print("=" * 60)


# =============================================================================
# Test Stub
# =============================================================================

if __name__ == "__main__":
    print("This module defines metadata extraction only.")
