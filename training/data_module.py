import lightning as L
from torch.utils.data import  DataLoader
from processing.data_loading import SegmentationDataset
import pandas as pd
from processing.utils import read_yaml_file

class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, loader_config_path: str):
        super().__init__()
        self.loader_config = read_yaml_file(loader_config_path)

    def setup(self, stage: str):
        df_tr = pd.read_csv('data/train_df.csv')
        if stage == "fit":
            self.train_set = SegmentationDataset(samples=df_tr, mask_dir=self.loader_config['mask_dir'],
                                                 img_dir=self.loader_config['img_dir'],
                                                config_path= 'configs/processing.yaml', apply_transform=True, in_train_mode=True)

            self.val_set = SegmentationDataset(samples=df_tr, mask_dir=self.loader_config['mask_dir'],
                                                img_dir=self.loader_config['img_dir'],
                                                config_path='configs/processing.yaml', apply_transform=False, in_train_mode=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = SegmentationDataset(samples=df_tr,  img_dir=self.loader_config['img_dir'],
                                                mask_dir=self.loader_config['mask_dir'],
                                                config_path= 'configs/processing.yaml', apply_transform=False, in_train_mode=False)

        if stage == "predict":
            df_tst = pd.read_csv('data/test_df.csv')
            self.predict_set = SegmentationDataset(samples=df_tst, img_dir=self.loader_config['img_dir'],
                                                config_path= 'configs/processing.yaml', apply_transform=False, in_train_mode=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.loader_config['BATCH_SIZE'], shuffle=True,
                          num_workers=self.loader_config['NUM_WORKERS'])

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.loader_config['BATCH_SIZE'], shuffle=False,
                          num_workers=self.loader_config['NUM_WORKERS'])

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.loader_config['BATCH_SIZE'], shuffle=False,
                          num_workers=self.loader_config['NUM_WORKERS'])

    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_set, batch_size=self.loader_config['BATCH_SIZE'])


if __name__ == '__main__':
    data_module = SegmentationDataModule(loader_config_path='configs/trainer.yaml')
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    for batch in train_loader:
        print(batch['image'].shape, batch['mask'].shape)


    # for batch in val_loader:
    #     print(batch['image'].shape, batch['mask'].shape)
    #     break