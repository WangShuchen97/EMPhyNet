# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:10:24 2023

@author: Administrator
"""


import os
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torchviz import make_dot
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau,CyclicLR

from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP

from network.utils.tool import make_dir
from network.models.custom_losses import CustomSquareLoss,CustomSplitLoss

from network.models import FC
from network.models import RT
from network.models import RT_U_Net,U_Net,RT_withoutskip,RT_VAE,RT_DeepLab,RT_ViT

from torch.amp import autocast, GradScaler

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        

        
        #==============================================================
        
        networks_map = {
            'FC':FC.FC,
            'RT':RT.RT,
            'RT_U_Net':RT_U_Net.RT_U_Net,
            'U_Net':U_Net.U_Net,
            'RT_withoutskip':RT_withoutskip.RT_withoutskip,
            'RT_VAE':RT_VAE.RT_VAE,
            "RT_DeepLab":RT_DeepLab.RT_DeepLab,
            "RT_ViT":RT_ViT.RT_ViT
        }
        
        loss_map={
            'CrossEntropyLoss':nn.CrossEntropyLoss(),
            'MSELoss':nn.MSELoss(),
            'CustomSquareLoss':CustomSquareLoss(),
            'CustomSplitLoss':CustomSplitLoss()
            }
        
        #==============================================================
        
        if configs.model_name not in networks_map:
            raise ValueError('Name of model unknown %s' % configs.model_name)
        if configs.loss_function not in loss_map:
            raise ValueError('Name of loss function unknown %s' % configs.loss_function)
                
        Network = networks_map[configs.model_name]
        self.network = Network(configs).to(configs.device)
        self.loss_function=loss_map[configs.loss_function]
                
        
        if configs.is_amp:
            self.scaler = GradScaler(device='cuda') 
        if configs.is_ddp:
            self.network = DDP(self.network, device_ids=[int(self.configs.device[-1])],find_unused_parameters=True)
            
        self.set_optimizer()


    def set_optimizer(self,optimizer=None):
        if optimizer is None:
            self.optimizer = SGD(self.network.parameters(), lr=self.configs.learn_rate,weight_decay=self.configs.l2_weight_decay)
        else:
            self.optimizer = optimizer

        self.scheduler_CyclicLR = CyclicLR(self.optimizer, 
                    base_lr=self.configs.learn_rate, 
                    max_lr=self.configs.learn_rate*10, 
                    step_size_up=self.configs.learn_step_size_up,
                    mode='exp_range')
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode='min', 
                                           patience=self.configs.learn_rate_patience, 
                                           factor=self.configs.learn_rate_factor, 
                                           verbose=True,
                                           eps=0,
                                           cooldown=self.configs.learn_cooldown,
                                           min_lr=self.configs.learn_rate_min,
                                           threshold=self.configs.learn_threshold,
                                           threshold_mode=self.configs.learn_threshold_mode)
    

    def net_structure(self,input_size=(1,512,512),mode=None):
        if mode=="torchviz":
            x = torch.randn(input_size).to(self.configs.device)
            y = self.network(x)
            make_dot(y, params=dict(list(self.network.named_parameters()))).render("SimpleNet", format="png")
        if mode=="torchsummary":
            #summary(self.network, input_size=(self.configs.input_channel,self.configs.input_height,self.configs.input_width), batch_size=-1, device="cpu")
            summary(self.network, input_size=tuple(input_size[1:]), batch_size=-1, device='cuda')
        return
         
    def to_device(self,x):
        if type(x) == list:
            for i in range(len(x)):
                x[i]=torch.FloatTensor(x[i]).to(self.configs.device)
        else:
            x=torch.FloatTensor(x).to(self.configs.device)
        return x
        
    def train(self, sample, target,loss_function_parm=None):
        
        sample=self.to_device(sample)
        target=self.to_device(target)
        optimizer=self.optimizer
        optimizer.zero_grad(set_to_none=True)  
        if self.configs.is_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                output = self.network(sample)
                if loss_function_parm is None:
                    loss = self.loss_function(output, target)
                else:
                    loss = self.loss_function(output, target, loss_function_parm)

            self.scaler.scale(loss).backward()

            if self.configs.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.network.parameters(), self.configs.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            output=self.network(sample)
            if loss_function_parm is None:
                loss = self.loss_function(output, target)  
            else:
                loss = self.loss_function(output, target,loss_function_parm) 
            loss.backward()  

            if self.configs.max_grad_norm is not None:
                clip_grad_norm_(self.network.parameters(), self.configs.max_grad_norm)

            optimizer.step() 
            
        return loss
    
    def evaluate(self, sample, target,loss_function_parm=None):
        
        sample=self.to_device(sample)
        target=self.to_device(target)
        
        with torch.no_grad():
            output=self.network(sample)
        if loss_function_parm is None:
            loss = self.loss_function(output, target)  
        else:
            loss = self.loss_function(output, target,loss_function_parm)  
        return output.detach().cpu(),loss.detach().cpu()

    def save(self, save_name="model"):
        
        if self.configs.is_ddp:
            if self.configs.rank!=0:
                return
            
        stats = {'model_state_dict': self.network.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),}
        
        checkpoint_path = f"{self.configs.checkpoint_path}/{save_name}.pth"
        
        make_dir(self.configs.checkpoint_path)
        
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)
        return
        
    def load(self,load_name="model",load_param=None):
        if load_name is None:
            return
        checkpoint_path = f"{self.configs.checkpoint_path}/{load_name}.pth"
        
        params_before = {k: v.clone().detach() for k, v in self.network.state_dict().items()}

 
        if os.path.exists(checkpoint_path):
            stats = torch.load(checkpoint_path)
            #Ensure that the model and saved layer names are consistent for DDP and non parallel scenarios
            is_module_in_pth=False
            for key, _ in stats['model_state_dict'].items():
                
                if key.startswith('module.'):
                    is_module_in_pth=True
                    break
            is_module_in_model=False
            for key, _ in self.network.state_dict().items():
                if key.startswith('module.'):
                    is_module_in_model=True
                    break
            if is_module_in_pth:
                if is_module_in_model:
                    pass
                else:
                    stats['model_state_dict'] = {k.replace('module.', ''): v for k, v in stats['model_state_dict'].items()}
            else:
                if is_module_in_model:
                    stats['model_state_dict'] = {'module.'+k: v for k, v in stats['model_state_dict'].items()}
                else:
                    pass
            #===============================================

            if load_param is None:
                
                if self.configs.is_ddp:
                    self.network.load_state_dict(stats['model_state_dict'],strict=False)
                else:
                    self.network.load_state_dict(stats['model_state_dict'],strict=False)
            else:
                new_model_dict = self.network.state_dict()
                for param_name in load_param:
                    print(param_name)
                    new_model_dict[param_name] = stats['model_state_dict'][param_name]
                if self.configs.is_ddp:
                    self.network.load_state_dict(new_model_dict)
                else:
                    self.network.load_state_dict(new_model_dict)
            if (not self.configs.rank) or (self.configs.is_ddp and self.configs.rank==0):
                print("load model from %s" % checkpoint_path)
        else:
            print(f"The checkpoint file '{checkpoint_path}' does not exist.")

        params_after = self.network.state_dict()
        for k in params_before:
            if not torch.allclose(params_before[k], params_after[k]):
                pass
                #print(f"[OK] Parameter changed after load: {k}")
            else:
                pass
                #print(f"[WARNING] Parameter not changed: {k}")

        return
    
    


    



    
