#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=0-1:30
#SBATCH --output=out/%N-%j.out
#SBATCH --job-name=resnet18
#SBATCH --account=def-ehyangit

ls $SLURM_TMPDIR
time cp -rf ../dataset/imagenet.tar.gz $SLURM_TMPDIR && ls
cd $SLURM_TMPDIR
tar --use-compress-program=pigz -xf imagenet.tar.gz
cd /scratch/s875chen/JPEG-Inspired-Compression

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip list

GPU_ID=0 CUDA_VISIBLE_DEVICES=${GPU_ID} python3.9 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital --lr_step_size 1 --lr_gamma 0.98 --data_path "$SLURM_TMPDIR" --dataset "imagenet" --model "mobileNet_v2" --epoch 20 -Q_initial_QF 50 --JPEG_alpha 100  --JPEG_learning_rate 0.0001 --weight_lum 0.01 --weight_chrom 0.01 --num_workers 16 --data_path "$SLURM_TMPDIR/imagenet"
