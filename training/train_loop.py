import torch
import pytorch_lightning as pl
import numpy as np
import os
from processing.utils import read_yaml_file
from tqdm import tqdm
from torchmetrics.classification import Dice
from training.modelling.models import SegmentationModels
import torch
from torchmetrics.segmentation import MeanIoU
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
import wandb

_ = torch.manual_seed(0)



class FieldInstanceSegment(pl.LightningModule):
  def __init__(self, model:SegmentationModels, config_path:str):
    super().__init__()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    self.model = model
    self.config = read_yaml_file(config_path)
    self.results_dir = self.config['dir']+'/output_masks/'
    if not os.path.exists(self.results_dir):
      os.makedirs(self.results_dir)

    # self.criterion = Dice(average='micro' , ignore_index  = 0)

    self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True )),
            ("focal", 0.9, BinaryFocalLoss(alpha=0.7, gamma=2)),
        ]

    self.miou = MeanIoU(num_classes=2, per_class=False, include_background=False)

  def training_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    logits = self.model.forward(x)

    # indexed_preds = (logits > self.config['threshold']).int()
    # miou_score = self.miou(indexed_preds, ground_mask)

    total_loss = 0
    logs = {}
    for loss_name, weight, loss in self.losses:
        ls_mask = loss(logits, ground_mask)
        total_loss += weight * ls_mask
        logs[f"train_mask_{loss_name}"] = ls_mask


    self.log("total_train_loss", total_loss, on_step = True, on_epoch=True, prog_bar=True, logger=True)
    self.log("jaccard_train_loss", logs['train_mask_jaccard'], on_step = True, on_epoch=True, prog_bar=True, logger=True)
    self.log("focal_train_loss", logs['train_mask_focal'], on_step = True, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_miou", miou_score, on_step = True, on_epoch=True, prog_bar=True, logger=True)
    return {"loss": total_loss}

  def validation_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']

    logits = self.model.forward(x)

    # indexed_preds = (logits > self.config['threshold']).int()
    # miou_score = self.miou(indexed_preds, ground_mask)

    total_loss = 0
    logs = {}
    for loss_name, weight, loss in self.losses:
        ls_mask = loss(logits, ground_mask)
        total_loss += weight * ls_mask
        logs[f"val_mask_{loss_name}"] = ls_mask


    self.log("total_val_loss", total_loss, on_step = True, on_epoch=True, prog_bar=True, logger=True)
    self.log("jaccard_val_loss", logs['val_mask_jaccard'], on_step = True, on_epoch=True, prog_bar=True, logger=True)
    self.log("focal_val_loss", logs['val_mask_focal'], on_step = True, on_epoch=True, prog_bar=True, logger=True)
    # self.log("val_miou", miou_score, on_step = True, on_epoch=True, prog_bar=True, logger=True)



    # if batch_idx == 0:
    #     num_images_to_log = min(5, x.size(0))  # Ensure we log no more than available images
    #     for i in range(num_images_to_log):
    #         # Convert tensors to numpy arrays and scale
    #         image = (x[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Convert to HWC format
    #         ground_mask_image = (ground_mask[i].cpu().numpy() * 255).astype(np.uint8)
    #         pred_mask_image = (logits[i].cpu().numpy() * 255).astype(np.uint8)

    #         # Log images with correct shapes
    #         self.logger.experiment.log({
    #             f'val_image_{i}': wandb.Image(image),
    #             f'val_ground_mask_{i}': wandb.Image(ground_mask_image),
    #             f'val_pred_mask_{i}': wandb.Image(pred_mask_image)
    #         })

    return {"loss": total_loss}

  def test_step(self, batch, batch_idx):
    x, ground_mask = batch['image'], batch['mask']
    logits = self.model.forward(x)

    # indexed_preds = (logits > self.config['threshold']).int()
    # miou_score = self.miou(indexed_preds, ground_mask)

    total_loss = 0
    logs = {}
    for loss_name, weight, loss in self.losses:
        ls_mask = loss(logits, ground_mask)
        total_loss += weight * ls_mask
        logs[f"test_mask_{loss_name}"] = ls_mask


    self.log("total_test_loss", total_loss, on_step = True, on_epoch=True, prog_bar=True, logger=True)
    self.log("jaccard_test_loss", logs['test_mask_jaccard'], on_step = True, on_epoch=True, prog_bar=True, logger=True)
    self.log("focal_test_loss", logs['test_mask_focal'], on_step = True, on_epoch=True, prog_bar=True, logger=True)
    # self.log("test_miou", miou_score, on_step = True, on_epoch=True, prog_bar=True, logger=True)
    return {"loss": total_loss}

  def predict_step(self, batch, batch_idx):

    x , batch_image_name = batch['image'], batch['image_name']
    pred_mask = self.model.forward(x).cpu().detach().numpy()

    # pred_mask = (pred_mask > self.config['threshold']).int()
    save_dir = self.results_dir + self.model.name
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    for i, name in enumerate(batch_image_name):
      np.save(f"{save_dir}/{name.split('.')[0]}.npy", pred_mask[i])
    return


  def configure_optimizers(self):
    optim =  torch.optim.Adam(params=self.model.model.parameters(), lr = self.config['lr'], weight_decay = self.config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.8, threshold=0.0005,
                                                              cooldown =2,verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,gamma = 0.995 ,last_epoch=-1,   verbose=True)

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'total_train_loss', 'name': 'lr_scheduler'}]

