import os
import numpy as np
from tifffile import imread, imsave
from tqdm import tqdm
from pydantic import BaseModel, DirectoryPath, PositiveInt

class Stitcher(BaseModel):
    patch_size: PositiveInt
    patch_offset: PositiveInt
    patch_img_dir: DirectoryPath
    patch_mask_dir: DirectoryPath
    save_img_dir: DirectoryPath
    save_mask_dir: DirectoryPath

    def stitch_image(self, patches, original_shape):
        stitched_image = np.zeros(original_shape, dtype=patches[0].dtype)
        patch_idx = 0
        for i in range(0, original_shape[0], self.patch_offset):
            for j in range(0, original_shape[1], self.patch_offset):
                if i + self.patch_size <= original_shape[0] and j + self.patch_size <= original_shape[1]:
                    stitched_image[i:i + self.patch_size, j:j + self.patch_size] = patches[patch_idx]
                    patch_idx += 1
        return stitched_image

    def load_patches(self, patch_dir, prefix, is_mask=False):
        patches = []
        format = 'npy' if is_mask else 'tif'
        patch_files = sorted([f for f in os.listdir(patch_dir) if f.startswith(prefix) and f.endswith(format)])
        for patch_file in patch_files:
            patch_path = os.path.join(patch_dir, patch_file)
            if is_mask:
                patches.append(np.load(patch_path))
            else:
                patches.append(imread(patch_path))
        return patches

    def stitch_images_and_masks(self, original_shapes):
        if not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir)
        if not os.path.exists(self.save_mask_dir):
            os.makedirs(self.save_mask_dir)

        for image_file, original_shape in tqdm(original_shapes.items(), desc="Stitching images and masks"):
            img_prefix = image_file.split('.')[0]

            image_patches = self.load_patches(self.patch_img_dir, img_prefix)
            mask_patches = self.load_patches(self.patch_mask_dir, img_prefix, is_mask=True)

            stitched_image = self.stitch_image(image_patches, original_shape)
            stitched_mask = self.stitch_image(mask_patches, original_shape[:2])

            imsave(os.path.join(self.save_img_dir, f"{img_prefix}_stitched.tif"), stitched_image)
            np.save(os.path.join(self.save_mask_dir, f"{img_prefix}_stitched.npy"), stitched_mask)
