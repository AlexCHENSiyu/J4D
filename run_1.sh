#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-2:00
#SBATCH --output=out/%N-%j.out
#SBATCH --account=def-ehyangit

# module load StdEnv/2020
# module load python/3.9
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements.txt

# python -m venv ~/newenv
# source ~/newenv/bin/activate

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_compression.py --JPEG_enable --freeze_model \
                                    --JPEG_alpha 10 --hardness 20.0 --JPEG_learning_rate 0.01 \
                                    --weight_lum 0.005 --weight_chrom 0.005 \
                                    --num_workers 16