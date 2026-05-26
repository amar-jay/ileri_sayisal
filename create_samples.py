import os
from matplotlib import pyplot as plt
import time 
import numpy as np
from PIL import Image
from dataset import LITSDataset, LITSImageTransform
import torch


# folder to save the samples
os.makedirs("dataset/images", exist_ok=True)

dataset = LITSDataset(
    images_dir="dataset/nii",
    masks_dir="dataset/nii",
    slice_axis=2,
    transform=LITSImageTransform(),
    mapping_cache_path = "slice_mapping.pkl",
    test_size=0.2,
    )

print(f"{dataset=}\n{len(dataset)=}")

dataset.set_split("test") # using the test split    
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


def save_image(sample, img_path):
    image = sample['image'][0] # fetch only first batch
    image = np.transpose(image.numpy(), (1, 2, 0))
    print(f"Saving image to path: {img_path}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join("dataset/images", img_path), bbox_inches='tight', pad_inches=0)
    #img = Image.fromarray(image)
    #img.save(os.path.join("dataset/images", img_path))

count = 0
for sample in dataloader:
    if count == 5:
        break
    save_image(sample, f"sample_{count}.png")
    count += 1
