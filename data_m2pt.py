# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:06:24 2023

@author: Administrator
"""

import urllib.request
import osmnx as ox
import os
import numpy as np
import h5py
import random
from PIL import Image
import matplotlib.pyplot as plt
from network.utils.tool import make_dir
import json
import torch
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor,as_completed
seed=50
random.seed(seed)

make_dir("./data")
input_folder = make_dir("./data/input")
output_rt_folder = make_dir("./data/output_rt_32")
output_real_folder = make_dir("./data/output_real_32")
output_imag_folder = make_dir("./data/output_imag_32")

path_map = './matlab/map_data'
path_map_height = './matlab/map_height'
path_data = './matlab/data'
extension = "pt"

delay_num=32
delay_interval=0.5e-8
delay=[delay_interval*i for i in range(delay_num)]



#Generate input from Building data
for i in os.listdir(path_map_height):
    if f"data_{i[4:-4]}.{extension}" in os.listdir(input_folder):
        continue 
    print(i)
    map_height=h5py.File(f"{path_map_height}/{i}",'r')
    map_height=map_height['map'][()]
    
    map_height=np.clip(map_height,0,255)
    map_height=torch.tensor(map_height, dtype=torch.float32)
    
    map_height=map_height.unsqueeze(0)
    torch.save(map_height, f"{input_folder}/data_{i[4:-4]}.{extension}")
    
    # image = Image.fromarray(np.uint8(map_height))
    # image=image.resize((128,128),Image.Resampling.LANCZOS)
    # image.save(f"{input_folder}/data_{i[4:-4]}.{extension}")


#Generate output from RT 
def output(i):

    name="data_"+os.listdir(input_folder)[i][5:-3]
    if name+".pt" in os.listdir(f"{output_real_folder}"):
        return i,'Exist'
    
    
    data=h5py.File(f"{path_data}/{name}.mat",'r')
    
    lon1, lat1, lon2, lat2 = map(float, name.split('_')[-1].split(','))
    
    lat_interval=(lat2-lat1)/(data['rays_save'].shape[0]+1)
    lon_interval=(lon2-lon1)/(data['rays_save'].shape[1]+1)
        
    fading_real = np.zeros((delay_num, data['rays_save'].shape[0], data['rays_save'].shape[1]))
    fading_imag = np.zeros((delay_num, data['rays_save'].shape[0], data['rays_save'].shape[1]))
    fading_rt = np.zeros((delay_num, data['rays_save'].shape[0], data['rays_save'].shape[1]))
    
    
    for j in range(data['rays_save'].shape[0]):
        for k in range(data['rays_save'].shape[1]):
            temp=data[data['rays_save'][j,k]]
            temp=json.loads(''.join([chr(c[0]) for c in temp]))
            temp_receive=[0 for _ in range(delay_num)]
            if temp=='NoRay' or temp=='NoRx':
                pass
            else:
                if type(temp['D']) is not list:

                    temp['D']=[temp['D']]
                    temp['L']=[temp['L']]
                    temp['P']=[temp['P']]
                    
                for a in range(len(temp['D'])):
                    
                    if temp['D'][a]//delay_interval<delay_num:
 
                        receive=temp['L'][a]*np.exp(1j*temp['P'][a])
                        
                        
                        receive=max(0,100-temp['L'][a])*np.exp(1j*temp['P'][a])
                        receive_real=np.real(receive)
                        receive_imag=np.imag(receive)
                        
                        for t in range(delay_num):
                            
                            # temp_x=0.5*np.pi*(delay[t]-temp['D'][a])/delay_interval
                            # if temp_x==0:
                            #     temp_x=1
                            # else:
                            #     temp_x=np.sin(temp_x)/(temp_x)
                            
                            temp_x=np.exp(-(delay[t]-temp['D'][a])**2/((delay_interval/1.5)**2))
                            
                            fading_real[t][j][k]+=receive_real*temp_x
                            fading_imag[t][j][k]+=receive_imag*temp_x
                            fading_rt[t][j][k]+=1*temp_x
                            

    fading_real=np.array([fading_real])
    fading_imag=np.array([fading_imag])
    fading_rt=np.array([fading_rt])
    
    fading_real=torch.tensor(fading_real, dtype=torch.float32)
    fading_imag=torch.tensor(fading_imag, dtype=torch.float32)
    fading_rt=torch.tensor(fading_rt, dtype=torch.float32)
    torch.save(fading_real, output_real_folder+"/"+name+".pt")
    torch.save(fading_imag, output_imag_folder+"/"+name+".pt")
    torch.save(fading_rt, output_rt_folder+"/"+name+".pt")
    return  i,name


if __name__ == "__main__":
    
    #parallel
    
    
    
    n_files = len(os.listdir(input_folder))

    with ProcessPoolExecutor(max_workers=50) as executor:
        
        results = [executor.submit(output, i) for i in range(n_files)]
        
        # results = executor.map(task, range(n_files))

        for future in as_completed(results):
            res = future.result()
            if res:
                i, name = res
                print(f"Done {i}: {name}")
                
             