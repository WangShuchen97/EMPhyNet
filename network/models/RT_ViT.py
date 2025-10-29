# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Patch Embedding ----------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H/patch, W/patch
        H_patch, W_patch = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        return x, H_patch, W_patch

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------- Transformer Block ----------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), dim, drop)
    def forward(self, x):
        # x: B, N, C
        x_ = self.norm1(x)
        x_attn, _ = self.attn(x_, x_, x_)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------- ViT ----------------
class ViT(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, embed_dim=384, patch_size=16,
                 depth=6, num_heads=12, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_chans)  
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x, H_patch, W_patch = self.patch_embed(x)  # B, N, embed_dim
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)  # B, N, out_chans
        # reshape回 patch feature map
        x = x.transpose(1, 2).contiguous().view(B, -1, H_patch, W_patch)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


class FDTD(nn.Module):
    def __init__(self, configs,base_c=8):
        super(FDTD, self).__init__()
        
        self.configs=configs

        self.Map=ViT(1,1)
        self.A=ViT(2,1)
        self.c=ViT(2,1)

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
        self.p = ViT(1,1)

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

class RT_ViT(nn.Module):
    def __init__(self, configs):
        super(RT_ViT, self).__init__()
        
        self.configs=configs
        
        self.RayMap=FDTD(configs,base_c=128)

        if 'main' in self.configs.model_mode:
            self.Phase= Phase(configs,base_c=128)
            
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
