GPU_ID=4


teacher_list=(  
                # CUB 200
                "resnet18" \ 
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

JEPG_learning_rate=0.5
# JEPG_learning_rate=0.5
# alpha_learning_rate=0.01
alpha_learning_rate=0.5

for teacher in ${teacher_list[*]}; do
    for trial in 2; do
        # for JEPG_alpha in 0.5 1 5; do
        # for JEPG_alpha in 50000; do
        for JEPG_alpha in 1; do
            # CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar.py --model ${teacher} \
            # CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher.py --model ${teacher} \
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_compression.py --model ${teacher} \
                                        -t ${trial} \
                                        --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} --JEPG_alpha ${JEPG_alpha}  \
                                        --num_bit 2 \
                                        # --freeze_model
                                        # --seed 11 \
                                        # --alpha_learning_rate ${alpha_learning_rate} --JPEG_alpha_trainable \
                                        # --log_file "_alpha_per_q" \
                                        # --log_file "_alpha_per_channel" \      
        done                            
    done
done
