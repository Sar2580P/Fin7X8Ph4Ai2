from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd, os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        mask_dir, img_dir ,
        transform: A.Compose,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.mask_dir = mask_dir
        self.img_dir = img_dir

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # idx = idx % len(self.samples)

        image_name, mask_name = self.samples.loc[idx, 'img'] , self.samples.loc[idx, 'mask']
        image = rasterio.open(os.path.join(self.img_dir, image_name))
        mask = np.load(os.path.join(self.mask_dir, mask_name))

        # apply augmentations
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            img, mask = sample["image"], sample["mask"]
            mask = (mask > 0).astype(np.uint8)
            mask = torch.from_numpy(mask)
        
        img.close()
        return {
            "features": img,
            "masks": mask.float(),
        }
        
train_aug = A.Compose([
            A.LongestMaxSize(max_size=800, p=1),
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=A.cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
            ToTensorV2()
        ])

val_aug = A.Compose([
            A.LongestMaxSize(max_size=800, p=1),
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=A.cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
            ToTensorV2()
        ])