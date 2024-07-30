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
from PIL import Image


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
        with rasterio.open(f'data/images/train_{num}.tif') as dataset:
            height , width = dataset.height , dataset.width   # taking 5th channel
        
        create_mask_polygons(height=height, width = width , polygons=polygons.loc[:, 'geometry'], save_path=f'{dir}/train_mask{num}')
    
    return

def create_df(dir:str ='data' , is_train :bool = True):
    df = pd.DataFrame(columns = ['img', 'mask'])
    
    prefix = 'train' if is_train else 'test'
    for i in range(50):
        img = f'{prefix}_{i}.tif'
        mask = f'{prefix}_mask{i}.npy' if is_train else ''
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

   
      
if __name__ == '__main__':
  
    if not os.path.exists('data/train_df.csv'):
        create_df()
        create_df(is_train = False)
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




    