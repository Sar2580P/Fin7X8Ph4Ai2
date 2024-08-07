import segmentation_models_pytorch as smp
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Any, Dict, Callable
from processing.utils import read_yaml_file
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

class SegmentationModels(BaseModel):
    config: Dict[str, Any] = None
    config_path: str
    model: smp.UnetPlusPlus = None
    preproc_func: Callable = None
    name : str = None
    model_naam:str = None
    config: Dict[str, Any] = None

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ()

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
                        encoder_name=self.config['encoder_name'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=self.config['encoder_weights'],     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels= self.config['in_channels'],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=self.config['classes'],                      # model output channels (number of classes in your dataset)
                    )
        elif self.model_naam == 'FPN':
            self.name = f"{self.model_naam}__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"

            self.model = smp.FPN(**self.config)
        elif self.model_naam == 'DeepLabV3':
            self.name = f"{self.model_naam}__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"

            self.model = smp.DeepLabV3(**self.config)

        # print(self.model)

        self.preproc_func = get_preprocessing_fn(self.config['encoder_name'], self.config['encoder_weights'])

    config = ConfigDict(arbitrary_types_allowed=True)

    def forward(self, x)->torch.Tensor :
        # x = self.preproc_func(x)

        # output : torch.Size([batch_sz, num_classes, h, w])
        x = self.model(x)
        x = torch.sigmoid(x)  # Ensure it's between 0 and 1
        x = x.squeeze(1)
        return x


if __name__ == '__main__':
    unet = SegmentationModels(config_path='configs/unet_family.yaml')
    unet.get_model()

    input_ = torch.rand((1, 3, 224, 224))
    output = unet.forward(input_)
    print(output.shape)
    print(unet.name)