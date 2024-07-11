#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=0-2:00
#SBATCH --output=out/%N-%j.out
#SBATCH --job-name=resnet18
#SBATCH --account=def-ehyangit

mkdir $SLURM_TMPDIR/data/cub2011
ls $SLURM_TMPDIR/data
time cp -rf ./data/cub2011/CUB_200_2011.tgz $SLURM_TMPDIR/data && ls
cd $SLURM_TMPDIR/data
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

python3.9 train_compression.py --JPEG_enable --freeze_model --num_workers 16 \
                            --hardness 20.0 --JPEG_learning_rate 0.01 \
                            --weight_lum 0.005 --weight_chrom 0.005  \
                            --root $SLURM_TMPDIR/data/cub2011