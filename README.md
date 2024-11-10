# JPEG-Inspired-DL
This zip file is for reproducing the experimental results in our paper [*Learned JPEG Compression for DNN Vision*] submitted at CVPR 2025.

<p align="center">
<img src="./Diagrams/workflow.png" width=80% height=80% 
class="center">
</p>

## Installation
The repo is tested with Python 3.9, CUDA 11.7, and based on the 'requirements.txt' or 'requirements.yaml' file provided.

## Training J4D
For ImageNet training, please use [this bash file](./run_imagenet.sh).
or use this command for QF-based initilization: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=10427 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "mobilenet_v2" --num_workers 64 \
                                    --data_path "your_dataset_path" \
                                    --epochs 20 --trial 1 --batch_size 64 \
                                    --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                    --Q_initial_QF 10 --weight_lum 0.1 --weight_chrom 0.1 
```

Use this command for sensitivity-based initilization: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=10427 -nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
                                --dataset "imagenet" --model "Swin_T" --num_workers 64 \
                                --data_path "your_dataset_path" \
                                --epochs 20 --trial 1 --batch_size 64 \
                                --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                --Q_initial_Sensitivity 15 --weight_lum 0.01 --weight_chrom 0.01 \
                                --val_resize_size 232
```
which will save the rate-accuracy (R-A) results and optimized qtable to a corresponding folder under `./save/imagenet`.


### Sensitivity-Based Initialization of Quantization Tables
Based on the drived Senstivity, we have followed the following steps to initialize the Q-Tables for JPEG-DL.
```
factor = max(1/Y_sens)
q_y = constand / (Y_sens * factor)
q_cbcr = 2 * constand / ((Cb_sens + Cr_sens) * factor)
```

### The sensitivity plots for 4 models on ImageNet and 3 models on CUB-200-2011 are shown as follows, alongside their corresponding initialization 
<p align="center">
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/imagenet_mobilenet_v2_sensitivity.png" width="25%" height="25%" class="center">
        <figcaption>ImageNet (MobileNetV2)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/imagenet_Mnasnet_sensitivity.png" width="25%" height="25%" class="center">
        <figcaption>ImageNet (MnasNet)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/imagenet_Swin_T_sensitivity.png" width="25%" height="25%"class="center">
        <figcaption>Flowers (Swin_T)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/imagenet_ConvNeXt_tiny_sensitivity.png" width="25%" height="25%"class="center">
        <figcaption>ImageNet (ConvNeXt-tiny)</figcaption>
    </figure>
</p>


<p align="center">
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/cub200_resnet18_sensitivity.png" width="25%" height="25%" class="center">
        <figcaption>CUB-200-2011 (ResNet18)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/cub200_mobilenet_v2_sensitivity.png" width="25%" height="25%" class="center">
        <figcaption>CUB-200-2011 (MobileNetV2)</figcaption>
    </figure>
    <figure style="display: inline-block; margin: 6px;">
        <img src="./Diagrams/cub200_Mnasnet_sensitivity.png" width="25%" height="25%"class="center">
        <figcaption>CUB-200-2011 (MnasNet)</figcaption>
    </figure>
</p>


## Acknowledgements
This repo is based on the code given in [PyTorch](https://github.com/pytorch/vision/tree/main/references/classification#resnet) for ImageNet.
