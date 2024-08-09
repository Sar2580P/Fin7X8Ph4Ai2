from inference.dataloader import SegmentationDataModule
from training.modelling.models import SegmentationModels
from training.train_loop import FieldInstanceSegment
from pytorch_lightning import Trainer
import os
from training.callbacks import (checkpoint_callback, rich_progress_bar, 
                                rich_model_summary)
from processing.utils import read_yaml_file, logger
import torch
from inference.patching_stitching import ImagePatcher

torch.set_float32_matmul_precision('medium')  #  | 'high'
data_module = SegmentationDataModule(loader_config_path='configs/trainer.yaml')


model = SegmentationModels(config_path='configs/unet_family.yaml')
model.get_model()

segmentation_setup = FieldInstanceSegment(config_path='configs/trainer.yaml', model=model)

#_____________________________________________________________________________________________________________
training_config = read_yaml_file('configs/trainer.yaml')

checkpoint_callback.dirpath = os.path.join(training_config['dir'], 'ckpts', model.name)
checkpoint_callback.filename = training_config['ckpt_file_name']

#_____________________________________________________________________________________________________________



if __name__ == '__main__':
    torch.cuda.empty_cache()
    trainer = Trainer(callbacks=[checkpoint_callback, rich_progress_bar, rich_model_summary],
                    accelerator = 'gpu'
                    )
    
    patcher = ImagePatcher(patch_size=(256, 256), overlap=(50, 50) ,output_dir='data/inference')
    
    if not os.path.exists('data/inference/patches'):
        os.makedirs('data/inference/patches')    

        source_dir = 'data/3channel_images'
        logger.info('Creating patches from images...')
        patcher.save_patches(source_dir)
    
    
    logger.info('Predicting masks on patches...')
    data_module.setup(stage="predict")
    segmentation_setup.results_dir += 'tests/'
    trainer.predict(dataloaders=data_module.predict_dataloader(), model=segmentation_setup , ckpt_path='last')
    logger.info(f'Patch Results saved to : {segmentation_setup.results_dir}{model.name}')
    
    logger.info('Reconstructing image from patches...')
    patcher.mask_patch_dir = f"{segmentation_setup.results_dir}{model.name}/"
    patcher.reconstruct_save_dir = f"{training_config['dir']}/output_masks/reconstructed/{model.name}"
    patcher.reconstruct_image('data/inference', 'data/inference/patches_metadata.json')
    
    logger.info(f'Image Reconstructed and saved to : {patcher.reconstruct_save_dir}')
    logger.info('------ Done! ------')
    