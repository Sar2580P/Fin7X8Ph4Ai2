import lightning as L
from torch.utils.data import  DataLoader
from processing.data_loading import SegmentationDataset
import pandas as pd
from processing.utils import read_yaml_file

class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, img_dir: str, mask_dir: str, loader_config_path: str):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.loader_config = read_yaml_file(loader_config_path)['datamodule']

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            df_tr = pd.read_csv('data/train_df.csv')
            self.train_set = SegmentationDataset(samples=df_tr, mask_dir=self.mask_dir, img_dir=self.img_dir, 
                                                config_path= 'configs/processing.yaml', apply_transform=False, in_train_mode=True)
            
            self.val_set = SegmentationDataset(samples=df_tr, mask_dir=self.mask_dir, img_dir=self.img_dir, 
                                                config_path='configs/processing.yaml', apply_transform=False, in_train_mode=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            df_tst = pd.read_csv('data/test_df.csv')
            self.test_set = SegmentationDataset(samples=df_tst,  mask_dir=self.mask_dir, img_dir=self.img_dir, 
                                                config_path= 'configs/processing.yaml', apply_transform=False, in_train_mode=False)

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.loader_config['BATCH_SIZE'], shuffle=True, 
                          num_workers=self.loader_config['NUM_WORKERS'])

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.loader_config['BATCH_SIZE'], shuffle=False, 
                          num_workers=self.loader_config['NUM_WORKERS'])

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.loader_config['BATCH_SIZE'], shuffle=False, 
                          num_workers=self.loader_config['NUM_WORKERS'])

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)