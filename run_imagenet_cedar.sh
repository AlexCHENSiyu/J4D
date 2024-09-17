#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0-11:59
#SBATCH --output=out/%N-%j.txt
#SBATCH --job-name=imagenet
#SBATCH --account=def-ehyangit
nvidia-smi


module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

time cp -rf ../dataset/imagenet.tar.gz $SLURM_TMPDIR && ls
cd $SLURM_TMPDIR
time tar --use-compress-program=pigz -xf imagenet.tar.gz
cd /scratch/s875chen/JPEG-Inspired-Compression


# ----------------------------------------------------mobilenet_v2----------------------------------------------------#
# Q_initial_QF = (10 12 14 16 18 20 30 40 50 60 70 80 90 100)
torchrun --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "mobilenet_v3" --data_path "$SLURM_TMPDIR/imagenet" \
                                    --epoch 20 --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
                                    --weight_lum 20.0 --weight_chrom 20.0 --num_workers 24 \
                                    --trial 2 --batch_size 64

# Q_initial_Sensitivity = (1 10 20 30 40 50 60 70 80 90 100)
# Weight = (0.01 0.1 1 5 10)
# torchrun --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
#                                     --dataset "imagenet" --model "mobilenet_v2" --data_path "$SLURM_TMPDIR/imagenet" \
#                                     --epoch 20 --JPEG_alpha 100 --Q_initial_Sensitivity 50 --JPEG_learning_rate 0.001 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 \
#                                     --num_workers 24 --lr_step_size 1 --lr_gamma 0.98 \
#                                     --trial 1 --batch_size 64 

# ------------------------------------------------------Resnet18-------------------------------------------------------#
# torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                   --dataset "imagenet" --model "resnet18" --data_path "$SLURM_TMPDIR/imagenet" \
#                                   --epoch 15 --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
#                                   --weight_lum 0.1 --weight_chrom 0.1 --num_workers 24 \
#                                   --trial 1 --batch_size 64 --test_only

# torchrun --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "resnet18" --data_path "$SLURM_TMPDIR/imagenet" \
#                                     --epoch 10 --JPEG_alpha 100 --Q_initial_QF 100 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 24 \
#                                     --trial 1 --batch_size 64 --resume "./save/imagenet/resnet18/JPEG_lr_0.01_alpha_100.0_weight_0.1_0.1_QF_100/trial_1"

# ----------------------------------------------------ConvNeXt_tiny----------------------------------------------------#
# torchrun --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "ConvNeXt_tiny" --epoch 15 --data_path "$SLURM_TMPDIR/imagenet" \
#                                     --JPEG_alpha 100 --Q_initial_QF 100 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 32 \
#                                     --trial 1 --batch_size 64 \
#                                     --learning_rate 1e-3 --lr_scheduler cosineannealinglr \
#                                     --lr_warmup_epochs 5 --lr_warmup_method linear --auto_augment ta_wide --random_erase 0.1 \
#                                     --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 --weight_decay 0.05 --norm_weight_decay 0.0 \
#                                     --train_crop_size 176 --model_ema --val_resize_size 236 --ra_sampler --ra_reps 4
