#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=0-10:30
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


# Regnet400mf, Resnet18, mobilenet_v2, ConvNeXt_tiny
torchrun --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "ConvNeXt_tiny" --data_path "$SLURM_TMPDIR/imagenet" \
                                    --epoch 15 --JPEG_alpha 100 --Q_initial_QF 100 --JPEG_learning_rate 0.01 \
                                    --weight_lum 0.1 --weight_chrom 0.1 --num_workers 24 \
                                    --lr_step_size 1 --lr_gamma 0.98 \
                                    --trial 1 --batch_size 64


#torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                   --dataset "imagenet" --model "resnet18" --data_path "$SLURM_TMPDIR/imagenet" \
#                                   --epoch 15 --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
#                                   --weight_lum 0.1 --weight_chrom 0.1 --num_workers 24 \
#                                   --lr_step_size 1 --lr_gamma 0.98 \
#                                   --trial 1 --batch_size 64 --test_only


# torchrun --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "resnet18" --data_path "$SLURM_TMPDIR/imagenet" \
#                                     --epoch 10 --JPEG_alpha 100 --Q_initial_QF 100 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 24 \
#                                     --lr_step_size 1 --lr_gamma 0.98 \
#                                     --trial 1 --batch_size 64 --resume "./save/imagenet/resnet18/JPEG_lr_0.01_alpha_100.0_weight_0.1_0.1_QF_100/trial_1"