# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large

class DeepLabLight(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, pretrained=True):
        super().__init__()
        # --- Backbone ---
        backbone = mobilenet_v3_large(pretrained=pretrained)
        if in_channels != 3:
            backbone.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        
        self.backbone = backbone.features
        self.low_level_idx = 2  
        self.high_level_idx = -1 

        # --- ASPP ---
        self.aspp = ASPP(960, 256)  

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        low_level = None
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx == self.low_level_idx:
                low_level = x  

        x = self.aspp(x)
        x = F.interpolate(x, size=(x.shape[2]*32, x.shape[3]*32), mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.atrous2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.atrous3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.atrous4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        size = x.shape[2:]
        x1 = self.atrous1(x)
        x2 = self.atrous2(x)
        x3 = self.atrous3(x)
        x4 = self.atrous4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        x = self.project_bn(x)
        x = self.relu(x)
        return x


class FDTD(nn.Module):
    def __init__(self, configs,base_c=8):
        super(FDTD, self).__init__()
        
        self.configs=configs

        self.Map=DeepLabLight(1,1)
        self.A=DeepLabLight(2,1)
        self.c=DeepLabLight(2,1)

    def forward(self,m):
        temp = torch.zeros_like(m, device=self.configs.device)
        ht   = torch.zeros_like(m, device=self.configs.device)
        m=self.Map(m)
        out=[]

        for i in range(self.configs.train_channel):        
            temp=self.A(torch.cat([temp,m], dim=1))
            ht=torch.cat([temp,ht], dim=1)
            ht=self.c(ht)
            ht_out = ht.unsqueeze(2)
            out.append(ht_out)
        return out
    
class Phase(nn.Module):
    def __init__(self, configs, base_c=32):
        super(Phase, self).__init__()
        
        self.configs = configs
        self.p = DeepLabLight(1,1)

    def forward(self,x):
        out=[]            
        temp = torch.zeros_like(x[0][:, :, 0, :, :])
        temp=temp.unsqueeze(2)
        for i in range(self.configs.train_channel):
            temp=self.p(x[i][:, :,0, :, :])
            temp=temp.unsqueeze(2)
            out.append(temp)
        out=torch.cat(out, dim=2)
        return out

class RT_DeepLab(nn.Module):
    def __init__(self, configs):
        super(RT_DeepLab, self).__init__()
        
        self.configs=configs
        
        self.RayMap=FDTD(configs,base_c=32)

        if 'main' in self.configs.model_mode:
            self.Phase= Phase(configs,base_c=32)
            
    def forward(self, m):
        if m.device.type!=self.configs.device:
            m=m.to(self.configs.device)
        
        Ray=self.RayMap(m)

        
        if self.configs.model_mode=="RayMap": 
            out = torch.cat(Ray, dim=2)
            return out
        if 'main' in self.configs.model_mode:
            out=self.Phase(Ray)
        return out 
