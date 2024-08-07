import segmentation_models_pytorch as smp
from processing.utils import read_yaml_file
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import torch.nn as nn

class SegmentationModels(nn.Module):
    def __init__(self, config_path: str):
        super(SegmentationModels, self).__init__()
        self.config_path = config_path
        self.model = None
        self.preproc_func = None
        self.name = None
        self.model_naam = None
        self.config = None

        self.get_model()

    def get_model(self):
        self.config = read_yaml_file(self.config_path)
        self.model_naam = self.config['model_name']
        self.config = self.config[f'{self.model_naam}_config']

        if self.model_naam == 'UnetPlusPlus':
            self.name = f"{self.model_naam}__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"
            self.model = smp.UnetPlusPlus(**self.config)

        elif self.model_naam == 'Unet':
            self.name = f"{self.model_naam}__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"
            self.model = smp.Unet(**self.config)

        elif self.model_naam == 'FPN':
            self.name = f"{self.model_naam}__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"
            self.model = smp.FPN(
                encoder_name=self.config['encoder_name'],
                encoder_weights=self.config['encoder_weights'],
                in_channels=self.config['in_channels'],
                classes=self.config['classes'],
            )

        elif self.model_naam == 'DeepLabV3':
            self.name = f"{self.model_naam}__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"
            self.model = smp.DeepLabV3(**self.config)

        self.preproc_func = get_preprocessing_fn(self.config['encoder_name'], self.config['encoder_weights'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.preproc_func(x)  # Uncomment this line if you need preprocessing
        x = self.model(x)
        x = torch.sigmoid(x)  # Ensure output is between 0 and 1
        x = x.squeeze(1)
        return x


if __name__ == '__main__':
    unet = SegmentationModels(config_path='configs/unet_family.yaml')
    unet.get_model()

    input_ = torch.rand((1, 3, 224, 224))
    output = unet.forward(input_)
    print(output.shape)
    print(unet.name)