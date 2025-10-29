# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""

import torch
import torch.nn as nn
from network.layers.basic import BasicConv, Down, Up, Outc,UpAttention
    
class Denoise_Network(nn.Module):
    def __init__(self,in_channels, out_channels,base_c=32,mode="Normal"):
        super(Denoise_Network, self).__init__()

        self.mode=mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.base_c=base_c
        
        self.inc = BasicConv(in_channels, base_c * 1, mid_channels=base_c)
        self.down1 = Down(base_c * 1, base_c * 2)     
        self.down2 = Down(base_c * 2, base_c * 4)     
        self.down3 = Down(base_c * 4, base_c * 8)     
        self.down4 = Down(base_c * 8, base_c * 16)    
        
        if mode=="Normal":
            self.midc = BasicConv(base_c * 32, base_c * 16, mid_channels=base_c)

        if mode!="m":
            self.up1 = Up(base_c * 16, base_c * 8)
            self.up2 = Up(base_c * 8, base_c * 4)
            self.up3 = Up(base_c * 4, base_c * 2)
            self.up4 = Up(base_c * 2, base_c * 1)
            
            self.outc = Outc(base_c * 1, out_channels,mid_channels=base_c)


    def forward(self, x,m=None):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)    

        if self.mode=="m":
            return x4
        if self.mode=="Normal":
            x4= torch.cat([x4,m], dim=1)
            x4=self.midc(x4)
        
        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.up4(y, x0)

        y=self.outc(y)
        
        return y


class FDTD(nn.Module):
    def __init__(self, configs,base_c=8):
        super(FDTD, self).__init__()
        
        self.configs=configs
        self.Map=Denoise_Network(1,self.configs.train_channel,base_c,mode="c")
        self.c=Denoise_Network(self.configs.train_channel,self.configs.train_channel,base_c,mode="c")


    def forward(self,m):
        m=self.Map(m)

        Ray=self.c(m)

        return Ray


class Phase(nn.Module):
    def __init__(self, configs, base_c=32):
        super(Phase, self).__init__()
        
        self.configs = configs
        self.p=Denoise_Network(self.configs.train_channel,self.configs.train_channel,base_c,mode="c")

    def forward(self,x):
        out=self.p(x)
        return out
    
class U_Net(nn.Module):
    def __init__(self, configs):
        super(U_Net, self).__init__()
        
        self.configs=configs

        self.RayMap=FDTD(configs,base_c=32)
        if 'main' in self.configs.model_mode:
            self.Phase= Phase(configs,base_c=32)

        
            
    def forward(self, m):
        if m.device.type!=self.configs.device:
            m=m.to(self.configs.device)
        
        Ray=self.RayMap(m)
        if self.configs.model_mode=="RayMap": 
            return Ray.unsqueeze(1)
        if 'main' in self.configs.model_mode:
            out=self.Phase(Ray)
        return out.unsqueeze(1)
