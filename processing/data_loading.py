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
from processing.utils import read_yaml_file
from torchvision.transforms import functional as F
from PIL import Image

import numpy as np

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

    # if not (contains_nan or contains_inf):
    #     print("Image does not contain NaN or infinity values.")

# Example usage
# Assuming `image` is your NumPy array representing the image
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
        # check_nan_inf(image)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        # Clamp values to the range [0, 1]
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

        # Apply augmentations if specified
        if self.apply_transform and self.in_train_mode:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.train_transforms(image)

            torch.manual_seed(seed)
            mask = self.train_transforms(mask)

        # Apply post-transforms to the image
        image = self.post_transforms(image)

        # Normalize image
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        # Convert mask to tensor separately
        mask = torch.from_numpy(np.array(mask)).long()

        return {
            "image": image.float(),
            "mask": (mask > 0.2).int(),
            "image_name": image_name.split('.')[0]
        }

if __name__ == "__main__":
    # Define the paths and parameters
    img_dir = r"data/3channel_images"
    mask_dir = r"data/masks"
    config_path = r"configs/processing.yaml"

    # Load the samples dataframe
    samples = pd.read_csv(r"data/train_df.csv")

    # Create an instance of the SegmentationDataset
    dataset = SegmentationDataset(samples, img_dir, config_path, mask_dir)

    # Iterate over the dataset and print the results
    for idx in range(len(dataset)):
        data = dataset[idx]
        print(data)
