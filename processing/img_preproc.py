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


if __name__ == '__main__':
  
  if not os.path.exists('data/train_df.csv'):
      create_df()
      create_df(is_train = False)
      print('Dataframes created successfully')