# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, C1=3, C2=3, latent_dim=128, hidden_dims=None):

        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # ---------------- Encoder ----------------
        enc_layers = []
        in_ch = C1
        for h in hidden_dims:
            enc_layers.append(
                nn.Conv2d(in_ch, h, kernel_size=3, stride=2, padding=1, bias=False)
            )
            enc_layers.append(nn.BatchNorm2d(h))
            enc_layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = h
        self.encoder_conv = nn.Sequential(*enc_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened_size = hidden_dims[-1] * 4 * 4

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # ---------------- Decoder ----------------
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        dec_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        for i in range(len(hidden_dims_rev) - 1):
            in_ch = hidden_dims_rev[i]
            out_ch = hidden_dims_rev[i + 1]
            dec_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            )
            dec_layers.append(nn.BatchNorm2d(out_ch))
            dec_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.decoder_convtrans = nn.Sequential(*dec_layers)


        self.final_conv = nn.Conv2d(hidden_dims_rev[-1], C2, kernel_size=3, padding=1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  

    def encode(self, x):
        x_enc = self.encoder_conv(x)         
        x_enc = self.adaptive_pool(x_enc)    
        b, c, h, w = x_enc.shape
        x_flat = x_enc.view(b, -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

    def decode(self, z, output_size):

        b = z.size(0)
        x = self.fc_dec(z)
        
        last_ch = self.encoder_conv[-3].num_features if hasattr(self.encoder_conv[-3], 'num_features') else None
        
        flattened_size = x.shape[1]
        channels = flattened_size // (4 * 4)
        x = x.view(b, channels, 4, 4)

        x = self.decoder_convtrans(x)  
        x = self.final_conv(x)
        
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        """
        x: [B, C1, H, W]
        returns: recon_x [B, C2, H, W], mu, logvar
        """
        B, C, H, W = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, output_size=(H, W))
        return recon

class FDTD(nn.Module):
    def __init__(self, configs,base_c=8):
        super(FDTD, self).__init__()
        
        self.configs=configs

        self.Map=ConvVAE(1,1,base_c)
        self.A=ConvVAE(2,1,base_c)
        self.c=ConvVAE(2,1,base_c)

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
        self.p = ConvVAE(1,1,base_c)

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

class RT_VAE(nn.Module):
    def __init__(self, configs):
        super(RT_VAE, self).__init__()
        
        self.configs=configs
        
        self.RayMap=FDTD(configs,base_c=512)

        if 'main' in self.configs.model_mode:
            self.Phase= Phase(configs,base_c=512)
            
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
