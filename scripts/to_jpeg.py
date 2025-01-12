import os
import torch
import argparse
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
def save_image_tensor(image_tensor, output_dir, file_prefix, slice_idx, is_mask=False):
    """Saves a single tensor slice as a JPEG image."""
    os.makedirs(output_dir, exist_ok=True)
    # Normalize the image to [0, 255]
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min()) * 255
    image_array = image_tensor.numpy().astype('uint8')
    image_array = np.squeeze(image_array)
    
    # Convert to PIL Image and save
    image = Image.fromarray(image_array)
    suffix = "_mask" if is_mask else ""
    file_name = f"{file_prefix}_slice_{slice_idx:04d}{suffix}.jpeg"
    image.save(os.path.join(output_dir, file_name))

def process_dataset(dataset, output_dir, count=100, save_masks=False):
    """Processes a PyTorch dataset and saves images and masks as JPEG slices."""
    for i, data in tqdm(enumerate(dataset)):
        if i > count:
            break
        image = data['image']
        mask = data['mask']
        image_path = data['image_path']
        slice_idx = data['slice_idx']
        
        # Extract the base name for the image
        file_prefix = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save the image
        try:
            save_image_tensor(image, output_dir, file_prefix, slice_idx, is_mask=False)
        except Exception as e:
            print("Error trying to save {image_path} slice", e)
            continue
        # Optionally save the mask if it exists
        if save_masks and mask.numel() > 0:
            save_image_tensor(mask, output_dir, file_prefix, slice_idx, is_mask=True)
        
    print(f"Processed all slices ({len(dataset)=}) of {image_path} successfully")

if __name__ == "__main__":
    import sys
    import os

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from dataset import LITSDataset, LITSImageTransform
    parser = argparse.ArgumentParser(description="Convert a PyTorch dataset to JPEG slices.")
    parser.add_argument("--output_dir", default="../dataset/images", help="Path to the output directory for JPEG slices.")
    parser.add_argument("--save-masks", type=bool, default=True, help="Save masks as JPEG slices if available.")
    parser.add_argument("--count", type=int, default=100, help="Save masks as JPEG slices if available.")
    
    args = parser.parse_args()
    print(f"{args=}")

    print("might take a while...")
    dataset = LITSDataset(
        images_dir="../dataset/nii",
        masks_dir="../dataset/nii",
        slice_axis=2,
        transform=LITSImageTransform(),
        test_size=0.005,
        split="test")
    print("fetched dataset")
    process_dataset(dataset, args.output_dir, count=args.count, save_masks=args.save_masks)
