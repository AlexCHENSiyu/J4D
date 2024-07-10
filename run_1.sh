#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=16G
#SBATCH --time=2:00
#SBATCH --account=def-ehyangit

module load StdEnv/2020
module load python/3.9
virtualenv --no-download /scratch/s875chen/env
source /scratch/s875chen/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# python -m venv ~/newenv
# source ~/newenv/bin/activate

python train_compression.py --JPEG_enable --freeze_model \
                                    --JPEG_alpha 10 --hardness 20.0 --JPEG_learning_rate 0.01 \
                                    --weight_lum 0.005 --weight_chrom 0.005 \
                                    --num_workers 16