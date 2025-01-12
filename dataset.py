import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split


class LITSDataset(Dataset):
    """Dataset for converting 3D NIfTI volumes to 2D slices for semantic segmentation"""

    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        slice_axis: int = 2,  # 0: sagittal, 1: coronal, 2: axial
        transform: Optional[Callable] = None,
        test_size: float = 0.2,
        random_state: int = 21,
        split: str = "all",
        slice_filter: Optional[float] = 0.1,  # Filter empty/low-information slices
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.slice_axis = slice_axis
        self.transform = transform
        self.slice_filter = slice_filter

        # Get all image paths
        self.image_paths = sorted(list(self.images_dir.glob("volume-*.nii")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .nii files found in {images_dir}")

        # Pre-calculate valid slices for each volume
        slices_mapping = self._create_slices_mapping()

        # Split into train and test
        train_slices, test_slices = train_test_split(
            slices_mapping, test_size=test_size, random_state=random_state
        )
        if split == "train":
            self.slices_mapping = train_slices
        elif split == "test":
            self.slices_mapping = test_slices
        else:
            self.slices_mapping = slices_mapping

    def _get_slice(self, volume, slice_idx):
        """Get slice along specified axis"""
        if self.slice_axis == 0:
            return volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            return volume[:, slice_idx, :]
        elif self.slice_axis == 2:
            return volume[:, :, slice_idx]
        else:  # axis == 2
            raise Exception("unknown axis")

    def _create_slices_mapping(self) -> List[Tuple[Path, int]]:
        """Create mapping of valid slices for each volume"""
        mapping:List[Tuple[Path, int]] = []

        for img_path in self.image_paths:
            try:
                img = nib.load(str(img_path))
                volume = img.get_fdata()
                _mask = nib.load(str(img_path).replace("volume", "segmentation"))
                slice_volume = _mask.get_fdata()
            except Exception as e:
                print(f"Error in creating slices -{img_path}: {e}")
                continue
            # Get number of slices in the chosen axis
            n_slices = volume.shape[self.slice_axis]

            # For each slice, check if it contains enough information
            for slice_idx in range(n_slices):
                slice_2d = self._get_slice(slice_volume, slice_idx)

                # Check if slice contains enough non-zero pixels
                if self.slice_filter:
                    non_zero = np.count_nonzero(slice_2d) / slice_2d.size
                    if non_zero > self.slice_filter:
                        mapping.append((img_path, slice_idx))

        return mapping

    def __len__(self) -> int:
        return len(self.slices_mapping)

    def _load_slice(self, img_path: Path, slice_idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load a single 2D slice from the 3D volume"""
        # Load image
        img = nib.load(str(img_path))
        volume = img.get_fdata()

        # Extract slice
        slice_2d = self._get_slice(volume, slice_idx)

        # Normalize slice
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)

        # Convert to tensor
        slice_tensor = torch.from_numpy(slice_2d).float()
        slice_tensor = slice_tensor.unsqueeze(0)  # Add channel dimension

        # Load corresponding mask if available
        mask_tensor = None
        if self.masks_dir:
            mask_path = str(img_path).replace("volume", "segmentation")
            mask = nib.load(str(mask_path))
            mask_volume = mask.get_fdata()
            # try:
            #mask = nib.load(str(mask_path))
            #mask_volume = mask.get_fdata()
            # except:
            #     print(f"Error loading file: {mask_path}")
            #     # get a random image slice
            #     return self._load_slice(img_path, random.randint(0, volume.shape[self.slice_axis]-1))

            mask_2d = self._get_slice(mask_volume, slice_idx)

            mask_tensor = torch.from_numpy(mask_2d).long()

        return slice_tensor, mask_tensor

    def __getitem__(self, idx: int) -> dict:
        """Get a slice with its corresponding mask if available"""
        img_path, slice_idx = self.slices_mapping[idx]
        try:
            image, mask = self._load_slice(img_path, slice_idx)
            sample = {
                'image': image,
                'mask': mask if mask is not None else torch.tensor([]),
                'image_path': str(img_path),
                'slice_idx': slice_idx
            }

            if self.transform:
                sample = self.transform(sample)

            return sample
        except Exception as e:
            print(f"Error - {img_path}: {e}")
            # get a random image slice
            return self.__getitem__(random.randint(0, len(self)-1))




class LITSImageTransform:
    """
    Transform class for medical image segmentation.
    Includes intensity clipping, normalization, and optional data augmentation.
    """
    def __init__(
        self,
        intensity_clip: Tuple[float, float] = (0, 99.9),# Percentile values for intensity clipping
        augmentations: Optional[Callable] = None,# Augmentation function (e.g., Albumentations or TorchVision transforms)
        normalize: bool = True,
        rotation_range = 30,
        noise_factor = 0.02
    ):
        self.intensity_clip = intensity_clip
        self.augmentations = augmentations
        self.normalize = normalize
        self.rotation_range = rotation_range
        self.noise_factor = noise_factor

    def clip_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Clip intensity values using the specified percentiles."""
        lower_percentile, upper_percentile = np.percentile(image.numpy(), self.intensity_clip)
        return torch.clip(image, lower_percentile, upper_percentile)

    def normalize_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize intensity values to [0, 1]."""
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)

    def random_rotate(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly rotate the image and mask."""
        angle = random.uniform(-self.rotation_range, self.rotation_range)  # Random angle between [-rotation_range, rotation_range]
        image_rotated = TF.rotate(image, angle)
        mask_rotated = TF.rotate(mask, angle)
        return image_rotated, mask_rotated

    def add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        noise = torch.randn_like(image) * self.noise_factor
        noisy_image = image + noise
        return torch.clip(noisy_image, 0, 1)  # Ensure values remain in the valid range [0, 1]


    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        mask = sample['mask']

        # Intensity Clipping
        image = self.clip_intensity(image)

        # Normalize Intensity
        if self.normalize:
            image = self.normalize_intensity(image)

        # a bit of data transformation
        # image = self.add_noise(image)
        # random rotation
        # image, mask = self.random_rotate(image, mask)

        # Data Augmentation
        if self.augmentations:
            # Combine image and mask for consistent augmentation
            augmented = self.augmentations(image=image.numpy(), mask=mask.numpy())
            image = torch.from_numpy(augmented['image'])
            mask = torch.from_numpy(augmented['mask'])

        # Update sample
        sample['image'] = image
        sample['mask'] = mask
        return sample

if __name__ == "__main__":
    dataset = LITSDataset(
        images_dir="../dataset/nii",
        masks_dir="../dataset/nii",
        slice_axis=2,
        transform=LITSImageTransform(),
        test_size=0.2,
        split="train")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for sample in dataloader:
        print(sample.keys())
        break
    print("dataset testing...")
    print(f"{dataset=}")
    print("dataloader testing...")
    print(f"{dataloader=}")
