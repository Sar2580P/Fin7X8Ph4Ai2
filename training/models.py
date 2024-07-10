import segmentation_models_pytorch as smp
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Any, Dict, Callable
from processing.utils import read_yaml_file
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch

class UNet_Variants(BaseModel):
    config: Dict[str, Any] = None
    config_path: str
    model: smp.Unet = None
    preproc_func: Callable = None
    name : str = None

    def get_model(self):
        self.config = read_yaml_file(self.config_path)
        self.name = f"UNet__E-{self.config['encoder_name']}__W-{self.config['encoder_weights']}__C-{self.config['in_channels']}"
        
        self.model = smp.Unet(
                    encoder_name=self.config['encoder_name'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=self.config['encoder_weights'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels= self.config['in_channels'],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=self.config['classes'],                      # model output channels (number of classes in your dataset)
                )
        self.preproc_func = get_preprocessing_fn(self.config['encoder_name'], self.config['encoder_weights'])
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  
    
    def forward(self, x)->torch.Tensor :
        if self.config['apply_preprocessing']:
            x = self.preproc_func(x)
        
        # output : torch.Size([batch_sz, num_classes, h, w])
        return self.model(x)
    
    
if __name__ == '__main__':
    unet = UNet_Variants(config_path='configs/unet_family.yaml')
    unet.get_model()
    
    input_ = torch.rand((1, 3, 224, 224))
    output = unet.forward(input_)
    print(output.shape)
    print(unet.name)