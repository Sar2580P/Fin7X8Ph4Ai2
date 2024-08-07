import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, DirectoryPath, PositiveInt, field_validator, ValidationInfo
from tifffile import imread , imsave
from typing import List
import os
import cv2
import json

class Patch(BaseModel):
    source_image_dir: DirectoryPath
    source_mask_dir: DirectoryPath
    patch_size: PositiveInt
    save_patch_img_dir: str     # not initially present, DirectoryPath validates the path
    save_patch_mask_dir: str    # not initially present, DirectoryPath validates the path
    patch_offset: PositiveInt = 50

    # @field_validator('patch_offset')
    # def check_offset(cls, v, values, info: ValidationInfo):
    #     patch_size = values.get('patch_size', info.data.get('patch_size'))
    #     if patch_size is not None and v >= patch_size:
    #         raise ValueError('Offset must be less than patch size.')
    #     return v


    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def patchify_image(self, image):
        patches = []
        h, w = image.shape[:2]
        for i in range(0, h, self.patch_offset):
            for j in range(0, w, self.patch_offset):
                patch = image[i:min(i + self.patch_size, h), j:min(j + self.patch_size, w)]
                if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                    patches.append(patch)
        return patches

    def patchify_mask(self, mask):
        patches = []
        h, w = mask.shape[:2]
        for i in range(0, h, self.patch_offset):
            for j in range(0, w, self.patch_offset):
                patch = mask[i:min(i + self.patch_size, h), j:min(j + self.patch_size, w)]
                if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                    patches.append(patch)
        return patches

    def save_patches(self, patches, save_dir, prefix, is_mask=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        format = 'npy' if is_mask else 'tif'
        for idx, patch in enumerate(patches):
            save_path = os.path.join(save_dir, f"{prefix}-{idx}.{format}")
            if is_mask:
                np.save(save_path, patch)
            else:
                imsave(save_path, patch)

    def patchify_images_and_masks(self):
        if not os.path.exists(self.save_patch_img_dir):
            os.makedirs(self.save_patch_img_dir)
        if not os.path.exists(self.save_patch_mask_dir):
            os.makedirs(self.save_patch_mask_dir)

        train_image_files = [f for f in os.listdir(self.source_image_dir) if (f.endswith('.tif') and f.startswith('train'))]
        train_mask_files = [f for f in os.listdir(self.source_mask_dir) if f.endswith('.npy')]
        train_image_files.sort()
        train_mask_files.sort()

        for image_file, mask_file in tqdm(zip(train_image_files, train_mask_files),
                                          total=len(train_image_files), desc="Patchifying images and masks"):
            image_path = os.path.join(self.source_image_dir, image_file)
            mask_path = os.path.join(self.source_mask_dir, mask_file)

            image = imread(image_path)
            mask = np.load(mask_path)
            assert image.shape[:2] == mask.shape[:2], f"Image :{image.shape[:2]} and mask: {mask.shape[:2]} have different dimensions  {image_file} and {mask_file}"
            image_patches = self.patchify_image(image)
            mask_patches = self.patchify_mask(mask)

            assert len(image_patches) == len(mask_patches), "Number of image patches and mask patches must be equal"

            self.save_patches(image_patches, self.save_patch_img_dir, image_file.split('.')[0])
            self.save_patches(mask_patches, self.save_patch_mask_dir, mask_file.split('.')[0], is_mask=True)

    def create_patch_df(self, dir: str = 'data', is_train: bool = True):
        df = pd.DataFrame(columns=['img', 'mask', 'class label'])

        train_patched_images = [f for f in os.listdir(f'{dir}/patched_images') if f.startswith('train')]
        train_patched_masks = [f for f in os.listdir(f'{dir}/patched_masks') if f.startswith('train')]
        predict_patched_images = [f for f in os.listdir(f'{dir}/patched_images') if f.startswith('test')]

        train_patched_images.sort()
        train_patched_masks.sort()

        for i in range(len(train_patched_images)):
            img = train_patched_images[i]
            mask = train_patched_masks[i]
            label = int(img.split('_')[1].split('-')[0])
            df.loc[i] = [img, mask, label]

        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class label'], random_state=42)
        train_df, test_df = train_test_split(train_df, test_size=0.2, stratify=train_df['class label'], random_state=42)

        predict_df = pd.DataFrame(columns=['img', 'mask', 'class label'])
        for i in range(len(predict_patched_images)):
            img = predict_patched_images[i]
            predict_df.loc[i] = [img, '', '']

        predict_df.to_csv(f'{dir}/predict_patch_df.csv', index=False)
        train_df.to_csv(f'{dir}/train_patch_df.csv', index=False)
        val_df.to_csv(f'{dir}/val_patch_df.csv', index=False)
        test_df.to_csv(f'{dir}/test_patch_df.csv', index=False)
        return

    def get_bounding_box_XYWH_ABS(self, polygon_coords:List)->List:
        # Extract x and y coordinates
        x_coords = polygon_coords[0::2]
        y_coords = polygon_coords[1::2]

        # Compute the top-left corner (min x, min y)
        x_min = min(x_coords)
        y_min = min(y_coords)

        # Compute the width and height
        width = max(x_coords) - x_min
        height = max(y_coords) - y_min

        return [x_min, y_min, width, height]


    def generate_segmentation_json(self, csv_file_path, output_json_path):
        data = {"images": []}

        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        category_id = {'field'  : 0} # only one category

        # Iterate through each row in the DataFrame
        for index, row in tqdm(df.iterrows() , desc = 'Generating segmentation JSON for patches'):
            image_file = row['img']
            mask_file = row['mask']

            # Construct full paths for image and mask
            image_path = os.path.join(self.save_patch_img_dir, image_file)
            mask_path = os.path.join(self.save_patch_mask_dir, mask_file)

            if os.path.exists(mask_path):
                # Load the mask
                mask = np.load(mask_path)
                mask = mask.astype(np.uint8)  # Ensure mask is in uint8 format

                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Prepare annotations
                annotations = []
                for contour in contours:
                    # Get the x, y coordinates of the contour
                    segmentation = contour.flatten().tolist()
                    annotations.append({
                        "class": "field",  # Adjust class if needed
                        "segmentation": [segmentation] ,
                        'bounding_box': self.get_bounding_box_XYWH_ABS(segmentation) ,
                        'category_id': category_id['field']
                    })

                # Append image data to the main data structure
                data["images"].append({
                    "file_name": image_file,
                    "annotations": annotations ,
                    'height': mask.shape[0],
                    'width': mask.shape[1]
                })

        # Save the generated data to a JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
