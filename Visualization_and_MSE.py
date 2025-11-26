import os
import matplotlib.colors as mcolors
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
from network.utils.tool import make_dir



def nonlinear_alpha(data, vmin=None, vmax=None, alpha_min=0.1, alpha_max=1, power=4.0):
    """
    Generate a nonlinear transparency map based on the absolute value of the data.
    Set the transparency of NaN value to 0.
    
    parameter
    Data: np.ndarray or torch.Tensor, input data
    Vmin, vmax: Normalized range of transparency, automatically taking the minimum and maximum values of | data | by default
    Alpha_min, alpha_max: transparency upper and lower bounds
    Power: Control the non-linear strength (>1, the closer it is to the higher value, the more obvious it is)
    
    return:
    Alpha_map: Transparency matrix with the same shape as the data
    """
    if not isinstance(data, np.ndarray):
        data = data.cpu().numpy()  

    abs_data = np.abs(data)
    nan_mask = np.isnan(data)

    if vmin is None:
        vmin = np.nanmin(abs_data)
    if vmax is None:
        vmax = np.nanmax(abs_data)

    norm = (abs_data - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)

    norm_nonlinear = norm ** (1 / power)

    alpha_map = alpha_min + (alpha_max - alpha_min) * norm_nonlinear

    alpha_map[nan_mask] = 0.0

    return alpha_map

def Visualization(name,plot_i,plot_j,mode='rt'):
    num=32
    times=1
    num_samples=int(name.split('_')[1])
    
    if mode=='rt':
        output_dir_result='./results/results_output_rt_RT_32'
        output_dir_base='./data/output_rt_32' 
        times=100
    if mode=='real':
        output_dir_result='./results/results_output_real_RT_32'
        output_dir_base='./data/output_real_32'
    if mode=='imag':
    
        output_dir_result='./results/results_output_imag_RT_32'
        output_dir_base='./data/output_imag_32'



    colors = [
    "#375093", "#3A519A", "#3E56A1", "#425DA8", "#4664AF",
    "#4E70AF", "#5B7DB9", "#6890C2", "#7493CB", "#8096D4",
    "#8EA0DC", "#9EBCDB", "#AFC6DE", "#C0D0E1", "#C8D6E7",
    "#D4DDEB", "#E0E5EF", "#E8EDF1", "#F0E9E0", "#F2EBE5",
    "#EDD5B9", "#ECD0B4", "#E4B98F", "#DBA27D", "#D09268",
    "#C16D58", "#AD4A43", "#9B2F36", "#831A21"
    ]

    c = np.array([mcolors.to_rgb(col) for col in colors])
    
    n = 256
    x = np.linspace(0, 1, n)
    gamma = 0.3 
    
    x_centered = np.sign(x - 0.5) * np.abs(2*(x - 0.5))**gamma / 2 + 0.5
    
    rgb = np.zeros((n,3))
    for i in range(3):
        rgb[:,i] = np.interp(x_centered, np.linspace(0,1,len(c)), c[:,i])
    
    cmap = mcolors.ListedColormap(rgb)

    
    fading_result=torch.load(output_dir_result+'/'+name)
    fading_base=torch.load(output_dir_base+'/'+name)
    
    building_data=torch.load("./data/input/"+name)
    
    
    
    colors_building=["#000000","#D2B48C"]
    cmap_building = mcolors.LinearSegmentedColormap.from_list('my_cmap2', colors_building, N=256)
    
    
    file_path="./data/background_"+str(num_samples)+".png"
    if os.path.exists(file_path):
        background = mpimg.imread(file_path)
    else:
        Z_building = building_data[0,:, :]*100
        Z_building_masked = np.where(np.abs(Z_building) < 1, 10, Z_building)
        background=Z_building_masked[:,:]


 


    fig, ax = plt.subplots(dpi=1200)
    Z_building = building_data[0,:, :]*100
    Z_building_masked = np.where(np.abs(Z_building) < 1, np.nan, Z_building)
    ax.imshow(Z_building_masked, cmap=cmap_building,vmin=-100, vmax=100,alpha=0.6,extent=[0,1,0,1],aspect='equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)  
    ax.set_position([0, 0, 1, 1])  
    plt.show()
    
    
    
    fig, ax = plt.subplots(dpi=1200)
    ax.imshow(background, extent=[0,1,0,1], aspect='equal')
    Z_building = building_data[0,:, :]*100
    Z_building_masked = np.where(np.abs(Z_building) < 1, np.nan, Z_building)
    ax.imshow(Z_building_masked, cmap=cmap_building,vmin=-100, vmax=100,alpha=0.6,extent=[0,1,0,1],aspect='equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)  
    ax.set_position([0, 0, 1, 1])  
    plt.show()
    
    
    
    
    fig, ax = plt.subplots(dpi=1200)
    ax.imshow(background, extent=[0,1,0,1], aspect='equal')
    ax.scatter(plot_j/256, 1-plot_i/256, color='red', s=30, marker='o', zorder=5)
    Z_building = building_data[0,:, :]*100
    Z_building_masked = np.where(np.abs(Z_building) < 1, np.nan, Z_building)
    ax.imshow(Z_building_masked, cmap=cmap_building,vmin=-100, vmax=100,alpha=0.6,extent=[0,1,0,1],aspect='equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)  
    ax.set_position([0, 0, 1, 1])  
    plt.show()
    
    
    
    delay_num = 32
    delay_interval = 0.5e-8
    delay = [delay_interval * i for i in range(delay_num)] 
    
    for m in range(plot_i-1, plot_i+2):        # i-1, i, i+1
        for n in range(plot_j-1, plot_j+2):    # j-1, j, j+1
            plt.figure(dpi=1200, figsize=(6, 3))
            # plt.plot(delay,fading_base[0, :, m, n]* times/100,'-', alpha=0.8,color='Black', linewidth=1.8, label='Baseline')
            # plt.plot(delay,fading_result[0, :, m, n]* times/100,'--', alpha=0.8,color='red', linewidth=1.2, label='EMPhyNet')
            plt.plot(delay,fading_base[0, :, m, n]* times/100,'-', alpha=0.8,color='Black', linewidth=2, label='Baseline',
                     marker='x', markersize=6, markerfacecolor='none', markeredgewidth=1)
            plt.plot(delay,fading_result[0, :, m, n]* times/100,'--', alpha=0.8,color='red', linewidth=1.2, label='EMPhyNet',
                     marker='o', markersize=5, markerfacecolor='none', markeredgewidth=0.8)
    
            if 'rt' in output_dir_result:
                plt.ylim(-0.1, 1)
            else:
                plt.ylim(-0.5, 0.5)
            plt.legend(fontsize=8)
            plt.gca().tick_params(axis='both', direction='in', length=6, width=1)
            plt.show()
    
    
    
     
    for t in range(num):
        
        fig, ax = plt.subplots(dpi=1200)
        ax.imshow(background, extent=[0,1,0,1], aspect='equal')
    
        threshold = 2  
        Z_signal = fading_base[0, t, :, :] * times
        Z_masked = np.where(np.abs(Z_signal) < threshold, np.nan, Z_signal)
        
    
        alpha_signal = nonlinear_alpha(Z_masked)
    
        im = ax.imshow(Z_masked, cmap=cmap, vmin=-100, vmax=100,
                       alpha=1, extent=[0,1,0,1], aspect='equal')
        
    
        ax.axis('off')
    
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_alpha(0)  
        ax.set_position([0, 0, 1, 1])  
        plt.show()
    
            
            
    for t in range(num):
        
        fig, ax = plt.subplots(dpi=1200)
    
        ax.imshow(background, extent=[0,1,0,1], aspect='equal')
    
        threshold = 2  
        Z_signal = fading_result[0, t, :, :] * times
        Z_masked = np.where(np.abs(Z_signal) < threshold, np.nan, Z_signal)
        
    
        alpha_signal = nonlinear_alpha(Z_masked)
    
        im = ax.imshow(Z_masked, cmap=cmap, vmin=-100, vmax=100,
                       alpha=1, extent=[0,1,0,1], aspect='equal')
        
    
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_alpha(0)  
        ax.set_position([0, 0, 1, 1])  
        plt.show()
    
    
    
    error=0
    
    for t in range(num):
        fig, ax = plt.subplots(dpi=1200)
        
        ax.imshow(background, extent=[0,1,0,1], aspect='equal')
        
    
        threshold = 2 
        Z_signal = torch.abs((fading_base[0, t, :, :] - fading_result[0, t, :, :]) * times)
        Z_masked = np.where(np.abs(Z_signal) < threshold, np.nan, Z_signal)
        
    
        alpha_signal = nonlinear_alpha(Z_masked)
        
    
        ax.imshow(Z_masked, cmap=cmap, vmin=-100, vmax=100,
                  alpha=1, extent=[0,1,0,1], aspect='equal')
        
    
        error += torch.mean((fading_base[0, t, :, :] - fading_result[0, t, :, :])**2)
        
    
        ax.axis('off')
        
    
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_position([0, 0, 1, 1])
        fig.patch.set_alpha(0) 
        
        plt.show()
        


def MSE(in_dir,base_out_dir,out_dir):
    num=32
    dir='./results'
    make_dir(dir)


    in_dir_list=os.listdir(in_dir)
    base_out_dir_list=os.listdir(base_out_dir)
    out_dir_list=os.listdir(out_dir)
    

    NRMSE=0

    NRMSE_MASK=0

    NRMSE_per_img = np.zeros(32, dtype=np.float32)

    NRMSE_MASK_per_img = np.zeros(32, dtype=np.float32)
    
    if 'rt' in base_out_dir:
        threshold = 0.001
    else:
        threshold = 0.1


    max_value=float('-inf')
    min_value=float('inf')
    
    with tqdm(out_dir_list,desc="out_dir", total=len(out_dir_list)) as f:
        new_out_dir=dir+'/'+out_dir.split("/")[-1]+'_MSE'
        make_dir(new_out_dir)
        for i, _ in enumerate(f):
            f.set_postfix({"Filename": out_dir_list[i]})

            new_out_dir_temp=new_out_dir+'/'+out_dir_list[i].rsplit(".", 1)[0]
            make_dir(new_out_dir_temp)

            data_in=torch.load(in_dir+'/'+out_dir_list[i], weights_only=False)
            
            data=torch.load(out_dir+'/'+out_dir_list[i], weights_only=False)

            base_data=torch.load(base_out_dir+'/'+out_dir_list[i], weights_only=False)

            max_q = torch.quantile(base_data.flatten(), 0.98).item()
            min_q = torch.quantile(base_data.flatten(), 0.02).item()
            max_value = max(max_value, max_q)
            min_value = min(min_value, min_q)

            diff=(data-base_data)**2
            NRMSE+=(torch.mean(diff)).item()

            temp = torch.mean(diff, dim=(2, 3))
            temp = temp.squeeze(0).cpu().numpy()
            temp = np.cumsum(temp)
            for t in range(len(temp)):
                NRMSE_per_img[t] += temp[t]/(t+1)



            mask = (data.abs() > threshold) | (base_data.abs() > threshold)
            diff = (data - base_data)**2
            diff = diff*mask
            

            
            if mask.sum().item()!=0:
                NRMSE_MASK+=diff.sum().item() / mask.sum().item()
              
            numerator=diff.sum(dim=(2,3))
            denominator=mask.sum(dim=(2,3))
            
            denominator=denominator.squeeze(0).cpu().numpy()
            denominator=np.cumsum(denominator)
            
            numerator=numerator.squeeze(0).cpu().numpy()
            numerator=np.cumsum(numerator)
            for t in range(len(numerator)):
                if denominator[t]!=0:
                    NRMSE_MASK_per_img[t]+=numerator[t]/denominator[t]
            
    print(max_value,min_value)

    NRMSE=np.sqrt(NRMSE/len(out_dir_list))/(max_value-min_value)
    NRMSE_MASK=np.sqrt(NRMSE_MASK/len(out_dir_list))/(max_value-min_value)

    NRMSE_per_img=list(np.sqrt(NRMSE_per_img/len(out_dir_list))/(max_value-min_value))
    NRMSE_MASK_per_img=list(np.sqrt(NRMSE_MASK_per_img/len(out_dir_list))/(max_value-min_value))


    log_file = dir+"/"+"MSE.log"  

    log_file_2 = dir+"/"+"MSE_per_img.log"  

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("NRMSE     "+out_dir+"  "+str(NRMSE)+ "\n")
        f.write("NRMSE_MASK"+out_dir+"  "+str(NRMSE_MASK)+ "\n")
        
    with open(log_file_2, "a", encoding="utf-8") as f:    
        for t in range(32):
            f.write("NRMSE        "+str(t)+"     "+out_dir+"  "+str(NRMSE_per_img[t])+ "\n")
            f.write("NRMSE_MASK   "+str(t)+"     "+out_dir+"  "+str(NRMSE_MASK_per_img[t])+ "\n")

    return
    
    
if __name__ == "__main__":
    name="data_135_-0.305625,51.5263,-0.3049,51.52675.pt"
    plot_i=80
    plot_j=170
    Visualization(name,plot_i,plot_j,mode='rt')
    Visualization(name,plot_i,plot_j,mode='real')
    Visualization(name,plot_i,plot_j,mode='imag')
    
    
    name="data_1962_-0.28605,51.5398,-0.285325,51.54025.pt"
    plot_i=180
    plot_j=50
    Visualization(name,plot_i,plot_j,mode='rt')
    Visualization(name,plot_i,plot_j,mode='real')
    Visualization(name,plot_i,plot_j,mode='imag')
    

    for i in ['RT','RT_withoutskip','RT_U_Net','RT_VAE','RT_DeepLab','RT_ViT','U_Net']:
        in_dir='./data/input'
        base_out_dir='./data/output_rt_32'
        out_dir='./results/results_output_rt_'+i+'_32'
        MSE(in_dir,base_out_dir,out_dir)


        in_dir='./data/input'
        base_out_dir='./data/output_real_32'
        out_dir='./results/results_output_real_'+i+'_32'
        MSE(in_dir,base_out_dir,out_dir)

        in_dir='./data/input'
        base_out_dir='./data/output_imag_32'
        out_dir='./results/results_output_imag_'+i+'_32'
        MSE(in_dir,base_out_dir,out_dir)

        



