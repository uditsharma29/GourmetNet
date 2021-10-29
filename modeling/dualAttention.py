import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import pdb

class AttentionModules(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(AttentionModules, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            lower_level_inplanes = 512
            lowest_level_inplanes = 1024
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
                                       
                                       
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self._init_weight()
        
        self.conv_high_level_feat = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False), 
        BatchNorm(256), 
        nn.Sigmoid())
        
        self.conv_low_level_feat = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),)
        self.conv_lower_level_feat = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),)
        self.conv_lowest_level_feat = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),)
                                       
        self.mul_level_final_reduction = nn.Sequential(nn.Conv2d(768, 256, kernel_size=1, stride=1, bias=False), 
        BatchNorm(256), 
        nn.ReLU())
        
                                       
        self.conv_spatial_mask = nn.Sequential(
                                       nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU())
                                       
        self.maxpool_spatial_mask = nn.MaxPool3d((128, 1, 1))
        
        self.avgpool_spatial_mask = nn.AvgPool3d((128, 1, 1))
        
        self.conv_spatial_mask2 = nn.Sequential(
                                       nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, bias=False),
                                       BatchNorm(1),
                                       nn.Sigmoid())
                                       
        self.final_conv = nn.Sequential(
                                       nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(2048),
                                       nn.ReLU())
                                       
        self.conv1 = nn.Conv2d(512, num_classes, kernel_size=1, bias=False)
        
        self.conv_from_resnet = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        
        self.bn2 = BatchNorm(64)
        
        self.conv2 = nn.Conv2d(low_level_inplanes,64, 1, bias=False)
        self.conv2_lower = nn.Conv2d(lower_level_inplanes, 64, 1, bias=False)
        self.conv2_lowest = nn.Conv2d(lowest_level_inplanes, 64, 1, bias=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, x, low_level_feat):

        input = x
        #Channel Attention
        low_level_feat = self.conv_low_level_feat(low_level_feat)
        x = self.conv_from_resnet(x)        
        ch_mask = self.global_avg_pool(x)
        ch_mask = self.conv_high_level_feat(ch_mask)
        out_channel = torch.mul(low_level_feat, ch_mask)
        
        #Spatial attention
        mask_sp_init = self.conv_spatial_mask(low_level_feat)
        max_branch = self.maxpool_spatial_mask(mask_sp_init)
        avg_branch = self.avgpool_spatial_mask(mask_sp_init)
        merge_branches = torch.cat((max_branch, avg_branch), dim=1)
        mask_spatial = self.conv_spatial_mask2(merge_branches)
        
        upsample_x = F.interpolate(input, size=mask_spatial.size()[2:], mode='bilinear', align_corners=True)

        out_spatial = torch.mul(upsample_x, mask_spatial)

        return out_spatial, out_channel



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_attention_modules(num_classes, backbone, BatchNorm):
    return AttentionModules(num_classes, backbone, BatchNorm)