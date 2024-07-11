from training.data_module import SegmentationDataModule
from training.models import UNet_Variants
from training.train_loop import FieldInstanceSegment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from training.callbacks import early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary
from processing.utils import read_yaml_file

data_module = SegmentationDataModule(loader_config_path='configs/trainer.yaml')
model = UNet_Variants(config_path='configs/unet_family.yaml')
model.get_model()

segmentation_setup = FieldInstanceSegment(config_path='configs/trainer.yaml', model=model)

#_____________________________________________________________________________________________________________
training_config = read_yaml_file('configs/trainer.yaml')

checkpoint_callback.dirpath = os.path.join(training_config['dir'], 'ckpts', model.name)
checkpoint_callback.filename = training_config['ckpt_file_name']

run_name = f"lr-{training_config['lr']}__bs-{training_config['BATCH_SIZE']}__decay-{training_config['weight_decay']}"
wandb_logger = WandbLogger(project= f"{model.name}", name = training_config['ckpt_file_name'])
csv_logger = CSVLogger(training_config['dir']+f"/{model.name}/"+'/logs/'+  training_config['ckpt_file_name'])

#_____________________________________________________________________________________________________________
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary], 
                  accelerator = 'cpu' ,max_epochs=training_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])  
 
trainer.fit(model=segmentation_setup, datamodule=data_module)
trainer.test(datamodule=data_module)