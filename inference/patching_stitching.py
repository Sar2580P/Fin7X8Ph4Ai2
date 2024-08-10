import os
import numpy as np
import rasterio
import json
import math
from tqdm import tqdm
import pandas as pd
from tifffile import imread , imwrite

class ImagePatcher:
    def __init__(self, output_dir , patch_size=(256, 256), overlap=(50, 50)):
        self.patch_size = patch_size
        self.overlap = overlap
        self.json_data = {}
        self.mask_patch_dir = ''
        self.reconstruct_save_dir = ''

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_patches(self, images_dir):
        SAVE_DIR = os.path.join(self.output_dir, 'patches')
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)


        df = pd.DataFrame(columns=['img' , 'mask' , 'label'])
        image_files = [f for f in os.listdir(images_dir) if f.startswith('test_') and f.endswith('.tif')]
        for img_idx, img_filename in tqdm(enumerate(image_files) ,total= len(image_files) , desc = 'Patching Test Images'):
            patch_id = 0
            img_path = os.path.join(images_dir, img_filename)
            with rasterio.open(img_path) as src:
                img = src.read()
                img_height, img_width = img.shape[1:3]

                # Padding the image
                pad_height = (self.patch_size[1] - img_height % self.patch_size[1]) % self.patch_size[1]
                pad_width = (self.patch_size[0] - img_width % self.patch_size[0]) % self.patch_size[0]
                padded_img = np.pad(img, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')

                # Extract and save patches
                y_offset = self.patch_size[1] - self.overlap[1]
                x_offset = self.patch_size[0] - self.overlap[0]
                patch_paths = []
                for y in range(0, padded_img.shape[1] - self.patch_size[1] + 1, y_offset):
                    for x in range(0, padded_img.shape[2] - self.patch_size[0] + 1, x_offset):
                        patch = padded_img[:, y:y+self.patch_size[1], x:x+self.patch_size[0]]
                        patch_filename = f"test_{img_idx}_{patch_id}.tif"
                        mask_filename = f"test_{img_idx}_{patch_id}.npy"
                        patch_path = os.path.join(SAVE_DIR, patch_filename)
                        imwrite(patch_path, patch)
                        df.loc[len(df)] = [patch_filename, mask_filename , -1]

                        patch_paths.append(patch_filename)

                        patch_id += 1

                if img_idx not in self.json_data:
                    self.json_data[img_idx] = []
                self.json_data[img_idx].append({
                    'image_path': patch_paths,
                    'original_h': img_height,
                    'original_w': img_width,
                    'h' : padded_img.shape[1],
                    'w' : padded_img.shape[2],
                    'img_filename': img_filename ,
                    'width_offset': x_offset,
                    'height_offset': y_offset ,
                })



        json_filename = os.path.join(self.output_dir, 'patches_metadata.json')
        with open(json_filename, 'w') as f:
            json.dump(self.json_data, f, indent=4)

        df.to_csv(os.path.join(self.output_dir, 'patches_metadata.csv'), index=False)

    def reconstruct_image(self, json_data_path, patch_shape=(256, 256)):
        # Read the JSON file
        with open(json_data_path) as f:
            json_data = json.load(f)

        if not os.path.exists(self.reconstruct_save_dir):
            os.makedirs(self.reconstruct_save_dir)

        assert os.path.exists(self.mask_patch_dir), f"mask_patch_dir: {self.mask_patch_dir} does not exist"

        for img_idx in tqdm(range(len(json_data)) , desc = 'Reconstructing Images'):
            meta_data = json_data[str(img_idx)][0]
            original_h, original_w = meta_data['original_h'], meta_data['original_w']
            padded_h, padded_w = meta_data['h'], meta_data['w']
            width_offset, height_offset = meta_data['width_offset'], meta_data['height_offset']
            patch_paths = meta_data['image_path']

            # Initialize the padded image with zeros
            padded_image = np.zeros((padded_h, padded_w))

            # Initialize the mask for counting overlaps
            overlap_mask = np.zeros((padded_h, padded_w))

            # Initialize position counters
            current_x , current_y = 0, 0
            patch_idx = 0

            while patch_idx < len(patch_paths):
                # Load patch
                patch_path = patch_paths[patch_idx]
                patch = np.load(os.path.join(self.mask_patch_dir, patch_path[:-4] + '.npy'))
                assert patch.shape == patch_shape, f"patch.shape: {patch.shape} patch_shape: {patch_shape}"

                # Compute end coordinates
                end_x = min(current_x + patch_shape[1], padded_w)
                end_y = min(current_y + patch_shape[0], padded_h)
                assert end_x <= padded_w and end_y <= padded_h, f"end_x: {end_x} padded_w: {padded_w} end_y: {end_y} padded_h: {padded_h}"

                # Compute overlapping region
                overlap_x = end_x - current_x
                overlap_y = end_y - current_y

                # Place patch in padded image and update the mask
                if overlap_x > 0 and overlap_y > 0:
                    padded_image[current_y:end_y, current_x:end_x] += patch[:overlap_y, :overlap_x]
                    overlap_mask[current_y:end_y, current_x:end_x] += 1

                # Update positions for the next patch
                current_x += width_offset

                # If we have reached the end of the current row, move to the next row
                if current_x >= padded_w:
                    current_x = 0
                    current_y += height_offset

                patch_idx += 1

            # Normalize by the number of overlaps
            # Avoid division by zero by ensuring mask values are not zero before division
            valid_mask = (overlap_mask > 0)
            padded_image[valid_mask] /= overlap_mask[valid_mask]

            # Trim the padded image to the original dimensions
            trimmed_image = padded_image[:original_h, :original_w]

            # Save the trimmed image
            np.save(os.path.join(self.reconstruct_save_dir, meta_data['img_filename'][:-4]), trimmed_image)

        return


if __name__ == '__main__':

    images_dir = 'data/3channel_images'
    output_dir = 'data/inference'
    patcher = ImagePatcher(patch_size=(256, 256), overlap=(50, 50) ,output_dir=output_dir)

    patcher.save_patches(images_dir)

    # # Later, when you need to reconstruct:
    # json_file = os.path.join(output_dir, 'patches_metadata.json')
    # reconstructed_images = patcher.reconstruct_image(output_dir, json_file)
