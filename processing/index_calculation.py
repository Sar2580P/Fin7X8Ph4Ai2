import os
import glob
from typing import Optional
import rasterio
import numpy as np
from pydantic import BaseModel, DirectoryPath, FilePath
from processing.utils import logger
from tqdm import tqdm

class IndexCalculationConfig(BaseModel):
    input_dir: DirectoryPath
    output_dir: str
    stack_indices: bool = True  # Whether to stack new channels or create a new image with only indices
    description: Optional[str] = "Generated using IndexCalculation tool"

    @classmethod
    def from_config(cls, config: dict) -> 'IndexCalculationConfig':        
        return cls(**config)

class IndexCalculator:
    def __init__(self, config: IndexCalculationConfig):
        self.config = config
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    def calculate_index(self, band1, band2):
        """Calculates the index using the formula (Band1 - Band2) / (Band1 + Band2)."""
        np.seterr(divide='ignore', invalid='ignore')
        index = (band1 - band2) / (band1 + band2)
        return np.nan_to_num(index, nan=-9999)

    def process_image(self, tif_path: FilePath):
        with rasterio.open(tif_path) as src:
            bands = src.read()
            profile = src.profile
            bands = np.nan_to_num(bands, nan=-9999)
            
            # Calculate indices
            ndvi = self.calculate_index(bands[7], bands[3])  # Using B8 (NIR) and B4 (Red)
            ndwi = self.calculate_index(bands[3], bands[10])  # Using B4 (Red) and B11 (SWIR)
            ndsi = self.calculate_index(bands[2], bands[10])  # Using B3 (Green) and B11 (SWIR)
            gsi = self.calculate_index(bands[3] - bands[1], bands[1] + bands[2] + bands[3])  # Using B4 (Red), B2 (Blue), B3 (Green)
            
            indices_stack = np.stack([ndvi, ndwi, ndsi, gsi], axis=0)

            if self.config.stack_indices:
                combined_stack = np.concatenate([bands, indices_stack], axis=0)
                new_descriptions = src.descriptions + ("NDVI", "NDWI", "NDSI", "GSI")
            else:
                combined_stack = indices_stack
                new_descriptions = ("NDVI", "NDWI", "NDSI", "GSI")
            
            # Update profile with the new number of bands and descriptions
            profile.update(count=combined_stack.shape[0], description=self.config.description)

            # Prepare the output path
            output_path = os.path.join(self.config.output_dir, os.path.basename(tif_path))
            
            # Write the new TIFF file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(combined_stack)
                dst.descriptions = new_descriptions

    def process_directory(self):
        tif_files = glob.glob(os.path.join(self.config.input_dir, "*.tif"))
        
        if not tif_files:
            logger.critical(f"No TIFF files found in {self.config.input_dir}")
            return
        
        for tif_file in tqdm(tif_files, desc="Generating indices from wavelengths"):
            try:
                self.process_image(tif_file)
            except Exception as e:
                logger.error(f"Error processing {tif_file}: {e}")


