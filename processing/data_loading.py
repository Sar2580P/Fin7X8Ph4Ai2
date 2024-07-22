from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd, os
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from processing.utils import read_yaml_file
from torchvision.transforms import functional as F

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
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
        ])
    
      # @property
    # def val_transforms(self):
    #     return A.Compose([
    #         A.LongestMaxSize(max_size=800, p=1),
    #         A.PadIfNeeded(min_height=800, min_width=800, border_mode=A.cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
    #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
    #     ])

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

        # Apply pre-transforms to the image and mask
        image = self.pre_transforms(image)
        mask = self.pre_transforms(mask)

        # Apply augmentations if specified
        if self.apply_transform and self.in_train_mode:
            # Convert image and mask to PIL for applying torchvision transformations
            image = F.to_pil_image(image)
            mask = F.to_pil_image(mask)
            
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.train_transforms(image)
            
            torch.manual_seed(seed)
            mask = self.train_transforms(mask)
            
            image = F.to_tensor(image)
            mask = F.to_tensor(mask)

        # Apply post-transforms to the image
        image = self.post_transforms(image)

        # Normalize image
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        # Convert mask to tensor separately
        mask = torch.from_numpy(np.array(mask)).long()

        return {
            "image": image.float(),
            "mask": mask,
            "image_name": image_name.split('.')[0]
        }




