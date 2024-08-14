import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from typing import Tuple
from tqdm import tqdm
from training.modelling.models import SegmentationModels
from training.train_loop import FieldInstanceSegment
from processing.utils import read_yaml_file, logger

class ImagePatcher:
    def __init__(self, inference_config_path : str):
        self.inf_config = read_yaml_file(inference_config_path)
        self.load_model()
    
    
    def load_model(self):
        model = SegmentationModels(config_path=self.inf_config['model_config'])
        model.get_model()
        segmentation_setup = FieldInstanceSegment.load_from_checkpoint(checkpoint_path=self.inf_config['ckpt'] ,config_path='configs/trainer.yaml', model=model)

        self.model = segmentation_setup.model.model
        logger.info(f"Loaded model : {model.name} from ckpt : {self.inf_config['ckpt']}")
        return


    def load_image(self, file_path: str) -> np.ndarray:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png']:
            with Image.open(file_path) as img:
                return np.array(img)
        
        elif file_ext in ['.tiff', '.tif']:
            return tifffile.imread(file_path)
        
        elif file_ext == '.npy':
            return np.load(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def extract_patches(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # Shape becomes (height, width, 1)
        h, w, c = image.shape
        ph, pw = self.inf_config['patch_size']
        
        pad_h = (ph - (h % ph)) % ph
        pad_w = (pw - (w % pw)) % pw
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        patches = []
        for i in range(0, padded_image.shape[0] - ph + 1, ph):
            for j in range(0, padded_image.shape[1] - pw + 1, pw):
                patch = padded_image[i:i+ph, j:j+pw, :]
                patches.append(patch)
        
        return torch.tensor(np.array(patches)).permute(0, 3, 1, 2)

    def plot_patches(self, patches: torch.Tensor, save_path: str):
        n_patches = patches.shape[0]
        n_cols = int(np.ceil(np.sqrt(n_patches)))
        n_rows = int(np.ceil(n_patches / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        for i, ax in enumerate(axes.flat):
            if i < n_patches:
                ax.imshow(patches[i].permute(1, 2, 0).numpy())
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def predict(self, patches:torch.Tensor):
        return self.model.forward(patches)

    def stitch_patches(self, patches: torch.Tensor, original_shape: Tuple[int, int, int]) -> np.ndarray:
        ph, pw = self.inf_config['patch_size']
        c = patches.shape[1]
        
        pad_h = (ph - (original_shape[0] % ph)) % ph
        pad_w = (pw - (original_shape[1] % pw)) % pw
        padded_shape = (original_shape[0] + pad_h, original_shape[1] + pad_w)
        
        recon_image = np.zeros((padded_shape[0], padded_shape[1], c), dtype=np.uint8)
        patch_idx = 0
        
        for i in range(0, padded_shape[0] - ph + 1, ph):
            for j in range(0, padded_shape[1] - pw + 1, pw):
                recon_image[i:i+ph, j:j+pw, :] = patches[patch_idx].permute(1, 2, 0).numpy()
                patch_idx += 1
        
        # trim the padding
        recon_image = recon_image[:original_shape[0], :original_shape[1], :]
        return recon_image.squeeze()

    def process_directory(self):
        if not os.path.isdir(self.inf_config['source_dir']):
            raise NotADirectoryError(f"Source directory not found: {self.inf_config['source_dir']}")
        
        if not os.path.isdir(self.inf_config['dest_dir']):
            os.makedirs(self.inf_config['dest_dir'])
        
        for file_name in tqdm(os.listdir(self.inf_config['source_dir']), 
                              desc = f'Inferencing on {self.inf_config['source_dir']}'):
            
            file_path = os.path.join(self.inf_config['source_dir'], file_name)
            
            image = self.load_image(file_path)
            patches = self.extract_patches(image)
            predicted_patches = self.predict(patches)
            stitched_image = self.stitch_patches(predicted_patches, image.shape)
            stitched_image_path = os.path.join(self.inf_config['dest_dir'], f"{os.path.splitext(file_name)[0]}_stitched")
            
            if self.inf_config['visualize']:
                # Plot patches and save
                plot_path = os.path.join(self.inf_config['dest_dir'], f"{os.path.splitext(file_name)[0]}_patches.png")
                self.plot_patches(patches, plot_path)
                        
                
                if stitched_image.ndim == 2:  # Grayscale
                    stitched_image = np.stack([stitched_image] * 3, axis=-1)  # Convert to RGB format for saving
                stitched_image_path += '.png'
                Image.fromarray(stitched_image*255).save(stitched_image_path)
            else : 
                # save as npy file
                stitched_image_path += '.npy'
                np.save(file=stitched_image_path , arr=stitched_image)



if __name__=='__main__':
    
    patcher = ImagePatcher(inference_config_path = 'inference_config_path')
    patcher.process_directory()
