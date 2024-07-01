import torch
import pytorch_lightning as pl
import numpy as np
import os
from processing.utils import read_yaml_file
from tqdm import tqdm

class FieldInstanceSegment(pl.LightningModule):
  def __init__(self, results_dir: str , model:torch.nn, config_path:str):
    super().__init__()
    self.model = model
    self.config = read_yaml_file(config_path)
    
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    self.results_dir = results_dir

    self.criterion = torch.nn.CrossEntropyLoss()

  
  def training_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    pred_mask = self.model.forward(x)
    loss = self.criterion(pred_mask, ground_mask)
    self.log("train_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)
    
    return loss
  
  def validation_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    pred_mask = self.model.forward(x)
    loss = self.criterion(pred_mask, ground_mask)
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    pred_mask = self.model.forward(x)
    loss = self.criterion(pred_mask, ground_mask)
    self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def predict_step(self, batch, batch_idx):
    x , batch_img_name = batch['image'], batch['image_name']
    pred_mask = self.model.forward(x).cpu().detach().numpy()
    
    for i, img_name in tqdm(enumerate(batch_img_name), desc='saving predicted mask'):
      np.save(f'{self.results_dir}/{img_name}.npy', pred_mask[i])
    return 

    
  def configure_optimizers(self):
    optim =  torch.optim.Adam(lr = self.config['lr'], weight_decay = self.config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.7, threshold=0.005, 
                                                              cooldown =2,verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,gamma = 0.995 ,last_epoch=-1,   verbose=True)
    
    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_loss', 'name': 'lr_scheduler'}]

