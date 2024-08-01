from typing import List
import tifffile as tiff
import numpy as np
import pandas as pd 
import geopandas as gpd
import cv2, os
from geopandas.geodataframe import GeoDataFrame
import rasterio
from tqdm import tqdm
from tifffile import imread, imwrite
import json


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


def create_mask_polygons(height: int, width: int, polygons: GeoDataFrame, save_path: str):
    # Ensure the save directory exists

    mask = np.zeros((height, width), dtype=np.uint8)
    for i, poly in enumerate(polygons):
      poly = np.array(poly.exterior.coords)
      poly = np.array([[x, y] for x, y in poly], dtype=np.int32)
      mask = cv2.fillPoly(mask, [poly], color=1)
        
    # Save the mask as a .npy file
    save_path = os.path.join(f'{save_path}.npy')
    np.save(save_path, mask)



def create_masks():
    with open("data/train_annotation.json") as f:
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
def update_annotations(data_dir:str = 'data/images', json_file:str = 'data/train_annotations.json', 
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
            bounding_box = get_bounding_box_XYWH_ABS(annotation['segmentation'])
            annotation['bounding_box'] = bounding_box
            annotation['category_id'] = category_id[annotation['class']]
            
    # save the updated json fle
    with open(save_path, 'w') as f:
        json.dump(data, f)    

def patchify_image(image, patch_size , offset = 50):
    patches = []
    h, w = image.shape[:2]
    for i in range(0, h, offset):
        for j in range(0, w, offset):
            patch = image[i:min(i+patch_size, h), j:min(j+patch_size, w)]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def patchify_mask(mask, patch_size ,offset = 50):
    patches = []
    h, w = mask.shape[:2]
    for i in range(0, h, offset):
        for j in range(0, w, offset):
            patch = mask[i:min(i+patch_size, h), j:min(j+patch_size, w)]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def save_patches(patches, save_dir, prefix , is_mask = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    format = 'npy' if is_mask else 'tif'
    for idx, patch in enumerate(patches):
        save_path = os.path.join(save_dir, f"{prefix}-{idx}.{format}")
        imwrite(save_path, patch)  

def patchify_images_and_masks(image_dir, mask_dir, patch_size, patched_image_dir, patched_mask_dir):
    if not os.path.exists(patched_image_dir):
        os.makedirs(patched_image_dir)
    if not os.path.exists(patched_mask_dir):
        os.makedirs(patched_mask_dir)

    train_image_files = [f for f in os.listdir(image_dir) if (f.endswith('.tif') and f.startswith('train'))]
    train_mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.npy')]
    train_image_files.sort()
    train_mask_files.sort()

    for image_file, mask_file in tqdm(zip(train_image_files, train_mask_files), total=len(train_image_files), desc="Patchifying images and masks"):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = imread(image_path)
        mask = np.load(mask_path)
        assert image.shape[:2] == mask.shape[:2], f"Image :{image.shape[:2]} and mask: {mask.shape[:2]} have different dimensions"
        image_patches = patchify_image(image, patch_size)
        mask_patches = patchify_mask(mask, patch_size)
        
        assert len(image_patches) == len(mask_patches), "Number of image patches and mask patches must be equal"

        save_patches(image_patches, patched_image_dir, image_file.split('.')[0])
        save_patches(mask_patches, patched_mask_dir, mask_file.split('.')[0])

if __name__ == '__main__':
    if not os.path.exists('data/train_df.csv'):
        create_df(is_train=True)
        create_df(is_train=False)
        print('Dataframes created successfully')
    
    if not os.path.exists('data/masks'):
        create_masks()
        print('Masks created successfully')
        
    if not os.path.exists('data/3channel_images'):
        apply_3_channel_preprocessing()
        print('3 channel images created successfully')
    
    if not os.path.exists('data/updated_train_annotation.json'):
        update_annotations()
        print('Updated annotations created successfully')

    if not os.path.exists('data/patched_images'):
        # Patchify images and masks
        image_dir = 'data/3channel_images'
        mask_dir = 'data/masks'
        patched_image_dir = 'data/patched_images'
        patched_mask_dir = 'data/patched_masks'
        patch_size = 256

        patchify_images_and_masks(image_dir, mask_dir, patch_size, 
                                  patched_image_dir, patched_mask_dir)
        print('Patchifying completed successfully')