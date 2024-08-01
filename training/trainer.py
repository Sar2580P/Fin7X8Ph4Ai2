from training.data_module import SegmentationDataModule
from training.modelling.models import SegmentationModels
from training.train_loop import FieldInstanceSegment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from training.callbacks import early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary
from processing.utils import read_yaml_file, logger
import wandb
from processing.plot_masks import plot_masks
import torch

torch.set_float32_matmul_precision('medium')  #  | 'high'
data_module = SegmentationDataModule(loader_config_path='configs/trainer.yaml')


model = SegmentationModels(config_path='configs/unet_family.yaml')
model.get_model()

segmentation_setup = FieldInstanceSegment(config_path='configs/trainer.yaml', model=model)

#_____________________________________________________________________________________________________________
training_config = read_yaml_file('configs/trainer.yaml')

checkpoint_callback.dirpath = os.path.join(training_config['dir'], 'ckpts', model.name)
checkpoint_callback.filename = training_config['ckpt_file_name']

run_name = f"lr-{training_config['lr']}__bs-{training_config['BATCH_SIZE']}__decay-{training_config['weight_decay']}"
wandb_logger = WandbLogger(project= f"{model.name}", name = training_config['ckpt_file_name'])
csv_logger = CSVLogger(training_config['dir']+f"/{model.name}/"+'/logs/'+  training_config['ckpt_file_name'])

def log_images(logger:WandbLogger):
    train_cols = ["caption", "sentinal-2_image", "ground_mask", "predicted_mask"]
    test_cols = ["caption", "sentinal-2_image", "predicted_mask"]
    train_data, test_data = [] , []

    for i in range(50):

        train_data.append([f"train_{i}.tif", wandb.Image(data_or_path = f"{training_config['img_dir']}/train_{i}.tif"),
                           wandb.Image(data_or_path = f"{training_config['mask_dir']}/train_mask{i}.npy"),
                           wandb.Image(data_or_path = f"results/output_masks/train_mask{i}.npy")])

        test_data.append([f"test_{i}.tif", wandb.Image(data_or_path = f"{training_config['img_dir']}/test_{i}.tif"),
                          wandb.Image(data_or_path = f"results/output_masks/test_mask{i}.npy")])

    logger.log_table(key="Training Samples", columns=train_cols, data=train_data)
    logger.log_table(key="Test Samples", columns=test_cols, data=test_data)

#_____________________________________________________________________________________________________________
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,max_epochs=training_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

data_module.setup(stage="fit")
trainer.fit(model = segmentation_setup , train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader())

data_module.setup(stage="test")
trainer.test(dataloaders=data_module.test_dataloader())

data_module.setup(stage="predict")
trainer.predict(dataloaders=data_module.predict_dataloader())
trainer.predict(dataloaders=data_module.val_dataloader() , ckpt_path='best')

#_____________________________________________________________________________________________________________

try:
    # log hyperparameters
    # wandb_logger.log_hyperparams(training_config)

    # log images
    log_images(wandb_logger)
except Exception as e:
    logger.error(f"Error logging to wandb: {e}")
    logger.info("Logging to wandb failed. Plotting masks locally.")
    plot_masks(train_dir=training_config['mask_dir'], predicted_dir='results/output_masks', ct=training_config['plot_masks'])