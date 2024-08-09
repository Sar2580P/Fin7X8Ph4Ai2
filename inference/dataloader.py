import lightning as L
from torch.utils.data import  DataLoader
from processing.data_loading import SegmentationDataset
from processing.utils import read_yaml_file

class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, loader_config_path: str):
        super().__init__()
        self.loader_config = read_yaml_file(loader_config_path)

    def setup(self, stage: str):
        if stage == "predict":
            self.predict_set = SegmentationDataset( samples='data/inference/patches_metadata.csv', is_patched_dataset=True,
                                                    img_dir= 'data/inference/patches',
                                                    config_path= 'configs/processing.yaml', apply_transform=False, in_train_mode=False,
                                                    in_predict_mode=True)

    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_set, batch_size=self.loader_config['BATCH_SIZE'] , shuffle=False,
                          num_workers=self.loader_config['NUM_WORKERS'])