GPU_ID=4


teacher_list=(  
                "resnet18"
				# "vgg8" \
                # "resnet20" \
                # "vgg13" \
				# "MobileNetV2"\
				# "wrn_40_1" \
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)

# JEPG_learning_rate: 0.0125 0.025 0.05 0.1
# JEPG_alpha: 0.02 0.04 0.06 0.08 0.1 0.2
# Q_inital: 1 16

JEPG_alpha=0.7
JEPG_learning_rate=0.3
block_size=8
Q_inital=1
num_bit=11
min_Q_Step=0

for teacher in ${teacher_list[*]}; do
    for trial in 1; do
        for JEPG_learning_rate in 0.1; do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 attack_teacher.py --model ${teacher} \
                                        --dataset "CUB200" --JPEG_enable \
                                        -t ${trial}  \
                                        --JEPG_alpha ${JEPG_alpha} \
                                        --block_size ${block_size} --Q_inital ${Q_inital} --num_bit ${num_bit} \
                                        --min_Q_Step ${min_Q_Step} --JEPG_learning_rate ${JEPG_learning_rate} \
                                        --log_file "_alpha_untrainable"
        done                            
    done
done
