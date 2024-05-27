GPU_ID=4



teacher_list=(  
				"vgg8" \
                # "resnet20" \
                # "vgg13" \
				# "MobileNetV2"\
				# "wrn_40_1" \
                # "wrn_40_2" \
                # "resnet32" \
                # "ShuffleV1" \
                # "ShuffleV2" \
				)


# vgg8_JEPG_alpha_0.6_JEPG_lr_0.3_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.6199951171875
# vgg8_JEPG_alpha_0.9_JEPG_lr_0.3_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.5199966430664
# vgg8_JEPG_alpha_0.5_JEPG_lr_0.3_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.27999877929688
# vgg8_JEPG_alpha_0.3_JEPG_lr_0.2_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.18999481201172
# vgg8_JEPG_alpha_2.0_JEPG_lr_0.15_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1	240	71.18999481201172
# vgg8_JEPG_alpha_0.4_JEPG_lr_0.3_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.16999816894531
# vgg8_JEPG_alpha_0.8_JEPG_lr_0.3_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.12999725341797
# vgg8_JEPG_alpha_0.6_JEPG_lr_0.2_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1    240 71.1199951171875
# vgg8_JEPG_alpha_0.6_JEPG_lr_0.15_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1	240	71.0199966430664
# vgg8_JEPG_alpha_0.7_JEPG_lr_0.15_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1	240	71.05999755859375
# vgg8_JEPG_alpha_0.4_JEPG_lr_0.15_Q_inital_1.0_Min_Q_0.0_alpha_untrainable_trial_1	240	71.00999450683594

JEPG_alpha=0.4
JEPG_learning_rate=0.15
block_size=8
Q_inital=1
num_bit=8
min_Q_Step=0

for teacher in ${teacher_list[*]}; do
    for trial in 2; do
        for JEPG_learning_rate in 0.2 0.15 0.1; do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_teacher_cifar.py --model ${teacher} \
                                        -t ${trial}  \
                                        --JEPG_alpha ${JEPG_alpha} \
                                        --block_size ${block_size} --Q_inital ${Q_inital} --num_bit ${num_bit} \
                                        --min_Q_Step ${min_Q_Step} --JEPG_learning_rate ${JEPG_learning_rate} \
                                        --JPEG_enable --log_file "_alpha_untrainable"
        done                            
    done
done
