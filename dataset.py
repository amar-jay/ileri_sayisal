import random
import os
import pickle
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
        mapping_cache_path = "slice_mapping.pkl", # Cache file path
        masks_dir: Optional[str] = None,
        slice_axis: int = 2,  # 0: sagittal, 1: coronal, 2: axial
        transform: Optional[Callable] = None,
        test_size: float = 0.2,
        random_state: int = 21,
        num_channels: int = 1, # should be 1 or 3 depending on the model's requirements. Errors are not so obvious 
        split: str = "all",
        slice_filter: Optional[float] = 0.05,  # Filter empty/low-information slices
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.slice_axis = slice_axis
        self.num_channels = num_channels
        self.transform = transform
        self.slice_filter = slice_filter
        self.split = split
        self.mapping_cache_path = mapping_cache_path

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
        # Check if the mapping is already cached
        if os.path.exists(self.mapping_cache_path):
            with open(self.mapping_cache_path, "rb") as f:
                print("Loading slice mapping from cache...")
                return pickle.load(f)

        print("Creating cache of slice mapping (may take a while)...")
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


        # Save the mapping for future use
        with open(self.mapping_cache_path, "wb") as f:
            print("Saving slice mapping to cache...")
            pickle.dump(mapping, f)

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
        if self.num_channels:
            slice_tensor = slice_tensor.repeat(self.num_channels, 1, 1)

        # Load corresponding mask if available
        mask_tensor = None
        if self.masks_dir:
            mask_path = str(img_path).replace("volume", "segmentation")

            try:
                mask = nib.load(str(mask_path))
                mask_volume = mask.get_fdata()
            except Exception as e:
                if self.split == "train":
                    print(f"Error loading file: {mask_path}")
                    # get a random image slice
                    return self._load_slice(img_path, random.randint(0, volume.shape[self.slice_axis]-1))
                else:
                    raise e
            mask_2d = self._get_slice(mask_volume, slice_idx)

            mask_tensor = torch.from_numpy(mask_2d)
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
            if self.split == 'train':
                print(f"Transform Error - {img_path}: {e} ")
                # get a random image slice
                return self.__getitem__(random.randint(0, len(self)-1))
            else:
                raise e




class LITSImageTransform:
    """
    Transform class for medical image segmentation.
    Includes intensity clipping, normalization, and optional data augmentation.
    """
    def __init__(
        self,
                train: bool = True,
        normalize: bool = True,
        output_size = [512, 512],
        augment_probability: float = 0.5,
        window_width: int = 400,
        window_level: int = 40,
        rotation_range = 30,
        noise_factor = 0.02,
        processor = None, #huggingface model processor
    ):
        self.train = train
        self.output_size = output_size
        self.augment_probability = augment_probability
        self.window_level = window_level
        self.window_width = window_width
        self.processor = processor

        self.normalize = normalize
        self.rotation_range = rotation_range
        self.noise_factor = noise_factor

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [-1, 1] range."""
        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())
            image = image * 2 - 1
        return image
    
    def window_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Apply window transform to CT image."""
        window_min = self.window_level - self.window_width // 2
        window_max = self.window_level + self.window_width // 2
        return torch.clamp(image, window_min, window_max)
    

    def random_rotate(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Apply random rotation."""
            if random.random() < self.augment_probability:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                if mask is not None:
                    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            return image, mask
    
    def random_flip(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply random horizontal/vertical flips."""
        if random.random() < self.augment_probability:
            if random.random() < 0.5:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                if mask is not None:
                    mask = TF.vflip(mask)
        return image, mask
    
    def random_gamma(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random gamma correction."""
        if random.random() < self.augment_probability:
            gamma = random.uniform(0.8, 1.2)
            image = TF.adjust_gamma(image, gamma)
        return image

    def random_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise."""
        if random.random() < self.augment_probability:
            noise = torch.randn_like(image) * 0.01
            image = image + noise
        return image

    def resize(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Resize image and mask to output size."""
        if image.shape[-2:] != self.output_size:
            image = TF.resize(image, self.output_size, 
                            interpolation=TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resize(mask, self.output_size,
                               interpolation=TF.InterpolationMode.NEAREST)
        return image, mask
    
    def normalize_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize intensity values to [0, 1]."""
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)
        
    def __call__(self, sample: dict) -> dict:
        image = sample['image']
        mask = sample['mask']


        # First resize to target size
        image, mask = self.resize(image, mask)
        # window transform
        image = self.window_transform(image)

        # Do Data Augmentation during training only
        if self.train:
            image = self.random_gamma(image)
            image = self.random_noise(image)
        
        image = self.normalize_intensity(image)
        image_pil = TF.to_pil_image(image.squeeze())
        mask_pil = TF.to_pil_image(mask.to(torch.uint8).squeeze()) if mask is not None else None

        
        image_pil, mask_pil = self.random_rotate(image_pil, mask_pil)
        image_pil, mask_pil = self.random_flip(image_pil, mask_pil)
        # # Normalize Intensity
        # if self.normalize:
        #     image = self.normalize_intensity(image)
        
        if self.processor:
            # Process for SAM
            inputs = self.processor(
                images=TF.to_tensor(image_pil),
                input_points=[[[450, 600]]] ,  # No prompt points for dense prediction
                return_tensors="pt"
            )
            # Update sample, if processor is used, pixel_values is initialized and image is dropped
            sample['pixel_values'] = inputs.pixel_values.squeeze()
            del sample['image'] # free the image to save memory
        else:
            sample['image'] = TF.to_tensor(image_pil)
            sample['mask'] = TF.to_tensor(mask_pil)

            
        # # Convert back to tensor for operations that work on tensors
        # image = TF.to_tensor(image_pil)
        # mask = TF.to_tensor(mask_pil) if mask_pil is not None else None
    

        return sample


if __name__ == "__main__":
    import time 

    start_time = time.time()

    dataset = LITSDataset(
        images_dir="dataset/nii",
        masks_dir="dataset/nii",
        slice_axis=2,
        transform=LITSImageTransform(),
        mapping_cache_path = "slice_mapping.pkl",
        test_size=0.2,
        split="test")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Record the time after loading the dataset
    dataset_load_time = time.time() - start_time

    for sample in dataloader:
        print(sample.keys())
        break
    print("dataset testing...")
    print(f"{dataset=}\n{len(dataset)=}")
    print("dataloader testing...")
    print(f"{dataloader=}")


    # Record the time after loading the dataset
    dataset_sample_time = time.time() - start_time - dataset_load_time
    print(f"TIME TO  LOAD  DATASET =  {dataset_load_time:.4f}s")
    print(f"TIME TO SAMPLE DATASET =  {dataset_sample_time:.4f}s")




