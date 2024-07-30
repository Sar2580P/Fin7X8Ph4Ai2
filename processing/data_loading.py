from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import os
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from utils import read_yaml_file
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
import random

def check_nan_inf(image):
    contains_nan = np.isnan(image).any()
    contains_pos_inf = np.isposinf(image).any()
    contains_neg_inf = np.isneginf(image).any()
    contains_inf = np.isinf(image).any()  # For both +inf and -inf

    if contains_nan:
        print("Image contains NaN values.")
    if contains_pos_inf:
        print("Image contains positive infinity values.")
    if contains_neg_inf:
        print("Image contains negative infinity values.")
    if contains_inf:
        print("Image contains infinity values (either positive or negative).")

class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        img_dir: str,
        config_path: str,
        mask_dir: str = None,
        apply_transform: bool = True,
        in_train_mode: bool = True
    ) -> None:
        self.samples = samples
        self.apply_transform = apply_transform
        self.in_train_mode = in_train_mode
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.config = read_yaml_file(config_path)



    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.6),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
        ])

    @property
    def post_transforms(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])

    @property
    def pre_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.config['img_sz'], self.config['img_sz']))
        ])

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name, mask_name = self.samples.loc[idx, 'img'], self.samples.loc[idx, 'mask']
        image = rasterio.open(os.path.join(self.img_dir, image_name)).read()
        image = np.transpose(image, (1, 2, 0))

        if self.mask_dir:
            mask = np.load(os.path.join(self.mask_dir, mask_name))
        else:  # for test data
            mask = np.zeros(image.shape[:-1], dtype=np.uint8)

        # Convert image and mask to uint8 before converting to PIL
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0, 1)
        mask = np.clip(mask, 0, 1)
        check_nan_inf(image)
        image = (image * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)

        # Convert image and mask to PIL images before applying pre-transforms
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Apply pre-transforms to the image and mask
        image = self.pre_transforms(image)
        mask = self.pre_transforms(mask)

        # Apply selection-based augmentation if specified


        # Apply post-transforms to the image
        image = self.post_transforms(image)

    
        # Convert mask to tensor separately
        mask = torch.from_numpy(np.array(mask)).long()

        return {
            "image": image.float(),
            "mask": (mask > 0.2).int(),
            "image_name": image_name.split('.')[0]
        }

def visualize_sample(dataset: SegmentationDataset, idx: int):
    sample = dataset[idx]
    image = sample["image"].permute(1, 2, 0).numpy()  # Convert CHW to HWC for plotting
    mask = sample["mask"].numpy()

    # Define a colormap for the mask
    cmap = mcolors.ListedColormap(['black', 'red'])  # Define colors: background (black), mask (red)
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap=cmap, norm=norm)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.show()

if __name__ == "__main__":
    # Define the paths and parameters
    img_dir = r"data/3channel_images"
    mask_dir = r"data/masks"
    config_path = r"configs/processing.yaml"
    save_dir = r"pics"

    # Load the samples dataframe
    samples = pd.read_csv(r"data/train_df.csv")

    # Create an instance of the SegmentationDataset
    dataset = SegmentationDataset(samples, img_dir, config_path, mask_dir)

    # Visualize a sample
    visualize_sample(dataset, idx=0)