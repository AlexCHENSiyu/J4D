#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-1:00
#SBATCH --output=out/%N-%j.out
#SBATCH --account=def-ehyangit

# module load StdEnv/2020
# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements.txt

teacher_list=(  
				# "vgg8" \
                "resnet18" \
                # "vgg13" \
				# "MobileNetV2"\
				# "wrn_40_1" \
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)

JEPG_alpha=0.4
JEPG_learning_rate=0.3
block_size=8
Q_inital=1
num_bit=8
min_Q_Step=1

for teacher in ${teacher_list[*]}; do
    for trial in 1; do
        for alpha_learning_rate in 0.3 0.2 0.1; do
            python train_compression.py --model ${teacher} \
            -t ${trial}  \
            --JEPG_alpha ${JEPG_alpha} \
            --block_size ${block_size} --Q_inital ${Q_inital} --num_bit ${num_bit} \
            --min_Q_Step ${min_Q_Step} --JEPG_learning_rate ${JEPG_learning_rate} \
            --JPEG_enable
        done                            
    done
done
