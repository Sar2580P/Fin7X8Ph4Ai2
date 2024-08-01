import torch
import pytorch_lightning as pl
import numpy as np
import os
from processing.utils import read_yaml_file
from tqdm import tqdm
from torchmetrics.classification import Dice
from training.modelling.models import SegmentationModels

class FieldInstanceSegment(pl.LightningModule):
  def __init__(self, model:SegmentationModels, config_path:str):
    super().__init__()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    self.model = model
    self.config = read_yaml_file(config_path)
    self.results_dir = self.config['dir']+'/output_masks'
    if not os.path.exists(self.results_dir):
      os.makedirs(self.results_dir)

    self.criterion = Dice(average='micro')


  def training_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    pred_mask = self.model.forward(x)
    print(f"Image min: {x.min()}, max: {x.max()}")
    print(f"Ground min: {ground_mask.min()}, max: {ground_mask.max()}")
    print(f"Pred min: {pred_mask.min()}, max: {pred_mask.max()}")
    # print(f"pred_mask dtype: {pred_mask.dtype}, shape: {pred_mask.shape}, requires_grad: {pred_mask.requires_grad}")
    # print(f"ground_mask dtype: {ground_mask.dtype}, shape: {ground_mask.shape}, requires_grad: {ground_mask.requires_grad}")
    loss = 1- self.criterion(pred_mask, ground_mask)
    self.log("train_loss", loss, on_step = False, on_epoch=True, prog_bar=True, logger=True)

    return loss.requires_grad_(requires_grad=True)

  def validation_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    pred_mask = self.model.forward(x)
    # print(f"pred_mask dtype: {pred_mask.dtype}, shape: {pred_mask.shape}, requires_grad: {pred_mask.requires_grad}")
    # print(f"ground_mask dtype: {ground_mask.dtype}, shape: {ground_mask.shape}, requires_grad: {ground_mask.requires_grad}")
    loss = 1- self.criterion(pred_mask, ground_mask)
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    pred_mask = self.model.forward(x)
    # print(f"pred_mask dtype: {pred_mask.dtype}, shape: {pred_mask.shape}, requires_grad: {pred_mask.requires_grad}")
    # print(f"ground_mask dtype: {ground_mask.dtype}, shape: {ground_mask.shape}, requires_grad: {ground_mask.requires_grad}")
    loss = 1- self.criterion(pred_mask, ground_mask)
    self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def predict_step(self, batch, batch_idx):
    x , batch_img_name = batch['image'], batch['image_name']
    pred_mask = self.model.forward(x).cpu().detach().numpy()

    for i, img_name in tqdm(enumerate(batch_img_name), desc='saving predicted mask'):
      np.save(f'{self.results_dir}/{img_name}.npy', pred_mask[i])
    return


  def configure_optimizers(self):
    optim =  torch.optim.Adam(params=self.model.model.parameters(), lr = self.config['lr'], weight_decay = self.config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.7, threshold=0.005,
                                                              cooldown =2,verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,gamma = 0.995 ,last_epoch=-1,   verbose=True)

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_loss', 'name': 'lr_scheduler'}]

