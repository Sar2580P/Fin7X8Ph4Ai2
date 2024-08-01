from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
from detectron2.config import CfgNode

from typing import List , Dict
import torch.nn as nn
import torch

class CustomDAFPN(FPN):
    """
    Custom FPN with an additional forward method.
    """

    def __init__(self, bottom_up: Backbone, in_features: List[str], out_channels: int):
        super().__init__(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            top_block=None,
            fuse_type="sum"
        )
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_feature in in_features:
            lateral_conv = nn.Conv2d(
                bottom_up.output_shape()[in_feature].channels, out_channels, kernel_size=1
            )
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): Input feature maps from the backbone.
        
        Returns:
            Dict[str, torch.Tensor]: Output feature maps from the FPN.
        """
        # Create lateral feature maps
        lateral_maps = [lateral_conv(x[f]) for f, lateral_conv in zip(self.in_features, self.lateral_convs)]
        
        # Build the top-down pathway
        fpn_maps = []
        prev_fpn_map = None
        for lateral_map in reversed(lateral_maps):
            if prev_fpn_map is None:
                fpn_map = lateral_map
            else:
                fpn_map = lateral_map + nn.functional.interpolate(prev_fpn_map, size=lateral_map.shape[-2:], mode='nearest')
            prev_fpn_map = fpn_map
            fpn_maps.append(fpn_map)
        
        fpn_maps = fpn_maps[::-1]  # Reverse the list to match the original order
        
        # Apply output convolutions
        output_maps = {f"p{i+2}": output_conv(fpn_map) for i, (fpn_map, output_conv) in enumerate(zip(fpn_maps, self.output_convs))}
        
        return output_maps

# Register the custom FPN
@BACKBONE_REGISTRY.register()
def build_custom_fpn_backbone(cfg: CfgNode, input_shape: ShapeSpec):
    bottom_up = BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.BOTTOM_UP)(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    return CustomDAFPN(bottom_up, in_features, out_channels)
