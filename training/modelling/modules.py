import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        '''
        Reference : Feature Pyramid with Double Attention
        '''
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, low_feature:torch.Tensor, high_feature:torch.Tensor ):

        high_feature_upsampled = F.interpolate(self.conv1x1(high_feature), size=low_feature.shape[2:], 
                                               mode='bilinear', align_corners=True)
        
        concat_feature = torch.cat([low_feature, high_feature_upsampled], dim=1)    # Concatenate along the channel dimension
        
        avg_pool = torch.mean(concat_feature, dim=1, keepdim=True)        # Global average pooling
        max_pool = torch.max(concat_feature, dim=1, keepdim=True)[0]   # Global max pooling
        pool_concat = torch.cat([avg_pool, max_pool], dim=1)   # Concatenate the pooled features along the channel dimension
        
        spatial_weight = self.conv7x7(pool_concat)        # Apply 7x7 convolution
        spatial_weight = self.sigmoid(spatial_weight)   # Apply sigmoid activation

        out = low_feature * spatial_weight    # Apply spatial weight to the original low_feature
        
        return out
    

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.conv1x1_high = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3_low = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, low_feature, high_feature):
        high_feature_conv = self.conv1x1_high(high_feature)
        low_feature_conv = self.conv3x3_low(low_feature)
        
        concat_feature = torch.cat([high_feature_conv, low_feature_conv], dim=1)   # Concatenate along the channel dimension
        
        # Global average pooling   Output shape : (batch_sz, in_channels)
        avg_pool = F.adaptive_avg_pool2d(concat_feature, 1).view(concat_feature.size(0), -1)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)
        
        # Global max pooling   Output shape : (batch_sz, in_channels)
        max_pool = F.adaptive_max_pool2d(concat_feature, 1).view(concat_feature.size(0), -1)
        max_pool = self.fc1(max_pool)
        max_pool = self.fc2(max_pool)
        
        # Pixel-wise sum of the pooled features
        channel_weight = avg_pool + max_pool
        
        # Apply sigmoid activation
        channel_weight = self.sigmoid(channel_weight).view(concat_feature.size(0), concat_feature.size(1), 1, 1)
        
        # Apply channel weight to the original low_feature
        out = low_feature * channel_weight
        
        return out
