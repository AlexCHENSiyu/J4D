#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=0-2:00
#SBATCH --output=out/%N-%j.out
#SBATCH --job-name=resnet18
#SBATCH --account=def-ehyangit

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip list

ls $SLURM_TMPDIR
time cp -rf ../dataset/imagenet.tar.gz $SLURM_TMPDIR && ls
cd $SLURM_TMPDIR
time tar --use-compress-program=pigz -xf imagenet.tar.gz
cd /scratch/s875chen/JPEG-Inspired-Compression

torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "resnet18" --data_path "$SLURM_TMPDIR/imagenet" \
                                    --epoch 5 --JPEG_alpha 100 --Q_initial_QF 40 --JPEG_learning_rate 0.01 \
                                    --weight_lum 1.0 --weight_chrom 1.0 --num_workers 24 --batch_size 64 \
                                    --lr_step_size 1 --lr_gamma 0.98 \
                                    --trial 1