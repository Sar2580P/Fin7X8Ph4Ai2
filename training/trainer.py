from training.data_module import SegmentationDataModule
from training.modelling.models import SegmentationModels
from training.train_loop import FieldInstanceSegment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from training.callbacks import (early_stop_callback, checkpoint_callback,
                                rich_progress_bar, rich_model_summary)
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

#_____________________________________________________________________________________________________________
torch.cuda.empty_cache()
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,max_epochs=training_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger] ,
                  accumulate_grad_batches=training_config['GRAD_ACCUMULATION_STEPS'])

data_module.setup(stage="fit")
trainer.fit(model = segmentation_setup , train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader() , ckpt_path='last')

data_module.setup(stage="test")
trainer.test(dataloaders=data_module.test_dataloader() , ckpt_path='last')


data_module.setup(stage="predict")
trainer.predict(dataloaders=data_module.predict_dataloader(), model=segmentation_setup , ckpt_path='last')

# ckpt_files = os.listdir(checkpoint_callback.dirpath)
# data_module.setup(stage="predict")
# model_name = model.name
# for ckpt_file in ckpt_files:
#     ckpt_path = os.path.join(checkpoint_callback.dirpath, ckpt_file)
#     model.name = model_name+ f"_{ckpt_file.split('_')[0]}"
#     trainer.predict(dataloaders=data_module.predict_dataloader(), model=segmentation_setup , ckpt_path=ckpt_path)
# model.name = model_name
#_____________________________________________________________________________________________________________

# checkpoint = torch.load('results/ckpts/DeepLabV3__E-resnet50__W-imagenet__C-3/Epoch-epoch=0__Loss-val_loss=0.00.ckpt')
# print(checkpoint['state_dict'].keys())