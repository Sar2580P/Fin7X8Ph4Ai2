from typing import List
import tifffile as tiff
import numpy as np
import pandas as pd
import geopandas as gpd
import cv2, os
from geopandas.geodataframe import GeoDataFrame
import rasterio
from tqdm import tqdm
import json
from processing.utils import read_yaml_file
from processing.patching import Patch
from processing.index_calculation import IndexCalculationConfig, IndexCalculator

def take_3_channels(channel_indices: List[int], img_path: str, save_path: str) -> np.ndarray:
    # Ensure exactly 3 channels are selected
    if len(channel_indices) != 3:
        raise ValueError("Exactly 3 channel indices must be provided")

    # Load the TIFF image
    img = tiff.imread(img_path)
    # Select the specified channels
    selected_channels = img[..., channel_indices]
    tiff.imwrite(save_path, selected_channels)
    return


def convert_to_geojson(data:dict, save_path:str = None):
  features = []
  for item in data:
    polygon = []
    for i in range(0, len(item['segmentation']), 2):
      polygon.append([item['segmentation'][i], item['segmentation'][i+1]])
    features.append({
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [polygon]
      },
      "properties": {"class": item['class']}
    })
  geojson_data =  { "type": "FeatureCollection", "features": features}
  polygons = gpd.GeoDataFrame.from_features(geojson_data,  crs="EPSG:4326")

  if save_path :
    polygons.to_file(save_path, driver="GeoJSON")
  return polygons

import numpy as np
import os
import cv2
from geopandas import GeoDataFrame

def create_mask_polygons(height: int, width: int, polygons: GeoDataFrame, save_path: str):
    # Ensure the save directory exists

    mask = np.zeros((height, width), dtype=np.uint8)
    for i, poly in enumerate(polygons.geometry):
        poly = np.array(poly.exterior.coords)
        poly = np.array([[x, y] for x, y in poly], dtype=np.int32)
        mask = cv2.fillPoly(mask, [poly], color=1)

    # # Create the complement mask
    # complement_mask = np.bitwise_xor(mask, 1)

    # # Stack the original and complement masks
    # mask = np.stack((complement_mask, mask), axis=-1)
    # assert mask.shape == (height, width, 2) , 'Mask shape is not correct'

    # Save the new mask as a .npy file
    save_path = os.path.join(f'{save_path}.npy')
    np.save(save_path, mask)






def create_masks():
    with open("data/train_annotations.json") as f:
        data = json.load(f)

    dir = 'data/masks'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # creating polygon geo_dataframe
    for num in tqdm(range(50), desc = 'creating masks for train images'):
        json_data = data["images"][num]["annotations"]
        geojson_data = convert_to_geojson(json_data)  # Convert to GeoJSON
        polygons = gpd.GeoDataFrame.from_features(geojson_data,  crs="EPSG:4326")
        # display(polygons)
        with rasterio.open(f'data/original_images/train_{num}.tif') as dataset:
            height , width = dataset.height , dataset.width   # taking 5th channel

        create_mask_polygons(height=height, width = width , polygons=polygons.loc[:, 'geometry'], save_path=f'{dir}/train_{num}')

    return

def create_df(dir:str ='data' , is_train :bool = True):
    df = pd.DataFrame(columns = ['img', 'mask'])

    prefix = 'train' if is_train else 'test'
    for i in range(50):
        img = f'{prefix}_{i}.tif'
        mask = f'{prefix}_{i}.npy' if is_train else ''
        df.loc[i] = [img, mask]

    df.to_csv(f'{dir}/{prefix}_df.csv', index = False)



def apply_3_channel_preprocessing():
    source_dir = 'data/images'
    output_dir = 'data/3channel_images'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    channel_indices = [4,5,6]
    for file_name in tqdm(os.listdir(source_dir),desc=f'Selecting channels: {channel_indices}'):
        if file_name.lower().endswith('.tif'):
            img_path = os.path.join(source_dir , file_name)
            # Process the image
            try:
                save_path = os.path.join(output_dir, file_name)
                take_3_channels(channel_indices=channel_indices, img_path = img_path, save_path= save_path)
                # print(f"Processed {img_path} and saved to {output_path}")
            except ValueError as e:
                print(f"Error processing {img_path}: {e}")

def get_bounding_box_XYWH_ABS(polygon_coords:List)->List:
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

# write function to load train_annoation.json, extract bounding box for
def update_annotations(data_dir:str = 'data/original_images', json_file:str = 'data/train_annotations.json',
                       save_path:str='data/updated_train_annotations.json'):
    '''
    This function takes the default annotation file and extracts various information from the images
    including the height, width, bounding box, and category_id. The updated information is saved to a new json file.
    '''

    with open(json_file) as f:
        data = json.load(f)

    category_id = {'field'  : 0} # only one category
    for img in tqdm(data['images'], desc='Updating annotations'):
        annotations = img['annotations']
        with rasterio.open(f"{data_dir}/{img['file_name']}") as dataset:
            height , width = dataset.height , dataset.width
        img['height'], img['width'] = height, width
        for annotation in annotations:
            annotation['bounding_box'] = get_bounding_box_XYWH_ABS(annotation['segmentation'])
            annotation['category_id'] = category_id[annotation['class']]
            annotation['segmentation'] = [annotation['segmentation']]

    # save the updated json fle
    with open(save_path, 'w') as f:
        json.dump(data, f)

def stack_masks_on_images(image_dir, mask_base_dir, save_dir):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    # List all mask directories
    mask_dirs = [os.path.join(mask_base_dir, d) for d in os.listdir(mask_base_dir) if os.path.isdir(os.path.join(mask_base_dir, d))]
    print(f"Found {len(mask_dirs)} mask directories")
    for image_file in tqdm(image_files , desc = "Stacking masks on images"):
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = tiff.imread(image_path)

        # Initialize a list to store the masks
        stacked_masks = []

        for mask_dir in mask_dirs:
            # Construct the mask file path
            mask_file = image_file.replace('.tif', '.npy')
            mask_path = os.path.join(mask_dir, mask_file)

            # Load the mask and append to the list
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                mask = np.expand_dims(mask, axis=-1)
                stacked_masks.append(mask)
            else:
                print(f"Mask {mask_path} not found.")

        # Stack the masks along with the image
        stacked_array = np.concatenate([image] + stacked_masks, axis=-1)

        # Save the stacked image-mask array
        save_path = os.path.join(save_dir, image_file)
        tiff.imwrite(save_path, stacked_array)





if __name__ == '__main__':
    processing_config = read_yaml_file('configs/processing.yaml')

    if not os.path.exists('data/train_df.csv'):
        create_df(is_train=True)
        create_df(is_train=False)
        print('Dataframes created successfully')

    if not os.path.exists('data/masks'):
        create_masks()
        print('Masks created successfully')

    # if not os.path.exists('data/3channel_images'):
    #     apply_3_channel_preprocessing()
    #     print('3 channel images created successfully')

    if not os.path.exists('data/updated_train_annotation.json'):
        update_annotations()
        print('Updated annotations created successfully')

    index_calculation_config = processing_config['index_calculation']
    if not os.path.exists(index_calculation_config['output_dir']):
        config = IndexCalculationConfig.from_config(index_calculation_config)

        index_calculator = IndexCalculator(config)
        index_calculator.process_directory()

    if not os.path.exists('data/patched_images'):
        patch_config = processing_config['patch_config']

        patch_maker = Patch.from_config(patch_config)
        patch_maker.patchify_images_and_masks()
        # patch_maker.patchify_images_only()
        patch_maker.create_patch_df(is_train=True)
        patch_maker.generate_segmentation_json('data/train_patch_df.csv', 'data/train_patch_annotation.json')
        patch_maker.generate_segmentation_json('data/val_patch_df.csv', 'data/val_patch_annotation.json')
        patch_maker.generate_segmentation_json('data/test_patch_df.csv', 'data/test_patch_annotation.json')
        print('Patchifying completed successfully')


    image_dir = "data/patched_images"
    mask_base_dir = "results/output_masks"  # This dir contains multiple mask dirs
    save_dir = "data/patched_images_stacked_masks"
    if not os.path.exists(save_dir):
        stack_masks_on_images(image_dir, mask_base_dir, save_dir)
        print('Stacked images created successfully')
