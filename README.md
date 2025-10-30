# Physics-Informed Deep Ray Tracing Network

## Introduction
XXXXXX

<br>
<div>
<img src="Figs/CIR_TO_Image.jpg" width="700px">
</div>
  <br>
<div>
<img src="Figs/Architecture.jpg" width="750px">
</div>
<br>

## Requirements

Linux + python>=3.8 + pytorch(GPU)

- matplotlib==3.7.5
- torch==2.4.1
- torchvision==0.19.1
- numpy==1.24.1
- h5py==3.11.0
- tqdm==4.67.1
- Pillow==10.2.0
- scikit-learn==1.3.2
- opencv-python==4.11.0.86

While the code is theoretically compatible with Windows, we highly recommend running it on a Linux system to ensure consistent results.

## Datasets and Pretrained models
- There are only two examples here, please download more data from [here](https://drive.google.com/drive/folders/1rOjZoe6gM9DRt03JC5UouguWeE6HedLi?usp=drive_link).

Building information obtained from [OpenStreetMap](https://www.openstreetmap.org/). The labels are constructed by [Ray Tracing of Matlab](https://www.mathworks.com/help/comm/ref/rfprop.raytracing.html). 

Please unzip `data.zip` and put the data `input`, `output_32`,and `output_overlap_32` in `./data` folder.

Please put Pretrained models `PIDRTN.pth`, `PIDRTN-A.pth`,and `U-net.pth` in `./results/checkpoints_ddp` folder.

## Run

Please refer to  `./run.sh` for training and testing.

Please refer to  `./ResultsVisualization.py` for visualization.

## Results

### Scenario 1

<div>
<img src="Figs/CIR1.jpg" width="700px">
</div>


### Scenario 2

<div>
<img src="Figs/CIR2.jpg" width="700px">
</div>

## Citation
Please cite our paper when you use this code.
```
XXX
```

## Contact
Please contact wangshuchen@ucas.ac.cn if you have any question about this work.
