from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd, os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from processing.utils import read_yaml_file

class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        img_dir:str ,
        config_path:str, 
        mask_dir:str = None, 
        apply_transform:bool = True,
        in_train_mode:bool = True
    ) -> None:
        self.samples = samples
        self.apply_transform = apply_transform
        self.in_train_mode = in_train_mode
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.config = read_yaml_file(config_path)
        
    @property
    def train_transforms(self):
        return A.Compose([
            A.LongestMaxSize(max_size=800, p=1),
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=A.cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
            ToTensorV2()
        ])
        
    # @property
    # def val_transforms(self):
    #     return A.Compose([
    #         A.LongestMaxSize(max_size=800, p=1),
    #         A.PadIfNeeded(min_height=800, min_width=800, border_mode=A.cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
    #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
    #         ToTensorV2()
    #     ])

    @property
    def post_transforms(self):
        return A.Compose([
            ToTensorV2()
        ])
        
    @property    
    def pre_transforms(self):
        return A.Compose([
                    A.Resize(self.config['img_sz'], self.config['img_sz'], p=1)
                ])

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # idx = idx % len(self.samples)

        image_name, mask_name = self.samples.loc[idx, 'img'] , self.samples.loc[idx, 'mask']
        image = rasterio.open(os.path.join(self.img_dir, image_name))
        
        if self.mask_dir :
            mask = np.load(os.path.join(self.mask_dir, mask_name))
        else : # for test data
            mask = np.zeros((image.height, image.width), dtype=np.uint8)

        sample = self.pre_transforms(image=image.read(), mask=mask)
        # apply augmentations
        if self.apply_transform:
            if self.in_train_mode:
                sample = self.train_transforms(image=sample['image'], mask=sample['mask'])
            # else:
            #     sample = self.val_transforms(image=sample['image'], mask=sample['mask'])
            
        # apply post transforms
        sample = self.post_transforms(image=sample['image'], mask=sample['mask'])
        
        # sample = self.transform(image=image, mask=mask)
        #     img, mask = , 
        #     mask = (mask > 0).astype(np.uint8)
        #     mask = torch.from_numpy(mask)
        image.close()
        return {
            "image": sample["image"],
            "masks": sample["mask"].float(),
            "image_name" : image_name.split('.')[0]
        }
    
    

