#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-16:00
#SBATCH --output=out/%N-%j.out
#SBATCH --account=def-ehyangit
module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install git+https://github.com/fra31/auto-attack
pip install git+https://github.com/RobustBench/robustbench.git
pip install git+https://github.com/ildoonet/pytorch-randaugment
python train-wa.py --desc preact-res18_trades_NCMI_1.😳.1_1.0_1 -m preact-resnet18 -na 200 --scheduler cosinew --NCMI --normalize True -param 1. 0.1 1. --tau .999 --beta 5.
python eval-aa.py --desc preact-res18_trades_NCMI_1.😳.1_1.0_1 -v custom