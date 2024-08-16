from pathlib import Path
from typing import List, Dict, Any, Tuple , Union
import pandas as pd
import os
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from processing.utils import read_yaml_file , logger
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
import random
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

def check_nan_inf(image:np.ndarray)->bool:
    contains_nan = np.isnan(image).any()
    contains_inf = np.isinf(image).any()  # For both +inf and -inf

    return contains_nan or contains_inf

class TransformBoth:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image, mask):
        seed = random.randint(0, 2**32)
        torch.manual_seed(seed)
        image = self.base_transform(image)
        torch.manual_seed(seed)
        mask = self.base_transform(mask)
        return image, mask

class PairCompose(transforms.Compose):
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: str,
        img_dir: str,
        config_path: str,
        is_patched_dataset:bool ,
        mask_dir: str = None,
        apply_transform: bool = True,
        in_train_mode: bool = True ,
        in_predict_mode: bool = False
    ) -> None:
        self.samples = pd.read_csv(samples)
        self.apply_transform = apply_transform
        self.in_train_mode = in_train_mode
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.config = read_yaml_file(config_path)
        self.is_patched_dataset = is_patched_dataset
        self.in_predict_mode = in_predict_mode

    @property
    def train_transforms(self):
        return PairCompose([
        TransformBoth(transforms.RandomRotation(degrees=4)),
        TransformBoth(transforms.RandomHorizontalFlip(p=0.5)),
        TransformBoth(transforms.RandomVerticalFlip(p=0.4)),
        TransformBoth(transforms.RandomAffine(degrees=2, translate=(0.01, 0.01), scale=(0.9, 1.1))),
        TransformBoth(AutoAugment(policy=AutoAugmentPolicy.CIFAR10)),
        ])

    @property
    def post_transforms(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])

    @property
    def pre_transforms(self):
        return transforms.Compose([
            # transforms.Resize((self.config['img_height'], self.config['img_width']))
            transforms.CenterCrop((self.config['img_height'], self.config['img_width']))
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

        # logger.info(f"min: {np.min(image)}, max: {np.max(image)}")
        # normalise image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Convert image and mask to uint8 before converting to PIL)
        if(check_nan_inf(image)):
            logger.critical(f"Image {image_name} contains NaN or Inf values")

            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0, 1)
            mask = np.clip(mask, 0, 1)
        image = (image*255 ).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)

        # Convert image and mask to PIL images before applying pre-transforms
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # mask  = np.transpose(mask, (2,0,1))
        if not self.is_patched_dataset:
            # Apply pre-transforms to the image and mask
            image = self.pre_transforms(image)
            mask = self.pre_transforms(mask)

        # Apply selection-based augmentation if specified
        if self.apply_transform and self.in_train_mode:
            image, mask = self.train_transforms(image, mask)

        # Apply post-transforms to the image
        image = self.post_transforms(image)

        '''
        normalising mask to range 0-1, by dividing by 255
        so that loss function does not give huge loss thus leading to high gradients
        '''
        mask = (np.array(mask)/255.0)> self.config['mask_threshold']
        mask = torch.from_numpy(mask).to(torch.int8)

        data=  {
            "image": image.float(),
            "mask": mask
        }
        if self.in_predict_mode:
            data.update({
                "image_name": image_name,
                "mask_name": mask_name.split('.')[0]
            })
        return data


if __name__ == "__main__":
    # Define the paths and parameters
    img_dir = r"data/3channel_images"
    mask_dir = r"data/patched_masks"
    config_path = r"configs/processing.yaml"
    save_dir = r"pics"

    # Load the samples dataframe
    samples = r"data/train_df.csv"

    # # Create an instance of the SegmentationDataset
    # dataset = SegmentationDataset(samples, img_dir, config_path, mask_dir)

    # idx = np.random.randint(0, 50)

    # augmented_sample = dataset.__getitem__(idx)
    # original_sample = {
    #     'image' : rasterio.open(os.path.join(img_dir, dataset.samples.iloc[idx , 0])).read()[2],
    #     'mask' : np.load(os.path.join(mask_dir, dataset.samples.iloc[idx , 1]))
    # }
    # print(original_sample['image'].shape)
    # print(augmented_sample['image'].shape)
    # # plot original vs augmented
    # fig, axes = plt.subplots(1, 4, figsize=(32, 6))

    # axes[0].imshow(original_sample['image'])
    # axes[0].set_title("Original Image")

    # axes[1].imshow(original_sample['mask'])
    # axes[1].set_title("Original Mask")

    # axes[2].imshow(augmented_sample['image'][1, :, :])
    # axes[2].set_title("Augmented Image")

    # axes[3].imshow(augmented_sample['mask'].numpy())
    # axes[3].set_title("Augmented Mask")
    # # super title
    # plt.suptitle(f"Original vs Augmented Image and Mask - {dataset.samples.iloc[idx, 0]}")

    # plt.savefig(os.path.join(save_dir, 'original_vs_augmented.png'))
#_______________________________________________________________________________________________________________________________
    img_dir = r"data/patched_images"
    mask_dir = r"data/patched_masks"
    config_path = r"configs/processing.yaml"
    save_dir = r"pics"

    # Load the samples dataframe
    samples = r"data/test_patch_df.csv"

    #checking image quality based on the threshold values
    # make a plot showing the masks created with different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fig , axes = plt.subplots(3, 3, figsize=(18, 18))

    idx = np.random.randint(0, 50)

    for num , threshold in enumerate(thresholds):
        dataset = SegmentationDataset(samples, img_dir, config_path,is_patched_dataset=True , mask_dir=mask_dir, apply_transform=False)
        dataset.config['mask_threshold'] = threshold

        sample = dataset.__getitem__(idx)

        #plot the mask
        axes[num//3, num%3].imshow((sample['mask']*255).numpy())

        axes[num//3, num%3].set_title(f"Threshold: {threshold}")
    plt.suptitle(f"Mask quality at different thresholds - {dataset.samples.iloc[idx, 0]}")
    plt.savefig(os.path.join(save_dir, 'mask_quality_VS_thresholds.png'))