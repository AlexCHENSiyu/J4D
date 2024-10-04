#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=0-0:59
#SBATCH --output=out/%N-%j.out
#SBATCH --job-name=resnet18
#SBATCH --account=def-ehyangit

ls $SLURM_TMPDIR
time cp -rf ./data/cub2011/CUB_200_2011.tgz $SLURM_TMPDIR && ls
cd $SLURM_TMPDIR
tar -xzf CUB_200_2011.tgz
rm CUB_200_2011.tgz
cd /scratch/s875chen/JPEG-Inspired-Compression

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip list

python3.9 train_cub200.py --JPEG_enable --freeze_model --QF_Inital --fixed_alpha \
                                --JPEG_alpha 10 --JPEG_learning_rate 0.01 \
                                --weight_lum 5.0 --weight_chrom 5.0 --epochs 10 \
                                --num_workers 16