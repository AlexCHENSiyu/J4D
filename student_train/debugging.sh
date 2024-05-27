GPU_ID=4

model=vgg13

student_list=(  
				"vgg8" \
				# "wrn_40_1"\
				# "ShuffleV1" \
				)

model_path=./save/models/${model}_vanilla/ckpt_epoch_240.pth

JEPG_learning_rate=0.5
# JEPG_learning_rate=0.5
# alpha_learning_rate=0.01
alpha_learning_rate=0.5

Temp=4
kl_coeff=0.9
ce_coeff=$(python -c "print(1.0-$kl_coeff)")

for student in ${student_list[*]}; do
    for trial in 1; do # 2 3
        for JEPG_alpha in 1; do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8  train_student.py --path_t ${model_path} \
                                        --num_workers 8  \
                                        --distill kd -r ${ce_coeff} -a ${kl_coeff} \
                                        --trial ${trial} --kd_T ${Temp} \
                                        --model_s ${student} \
                                        --trial ${trial} --seed 11 \
                                        --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} --JEPG_alpha ${JEPG_alpha}  \
                                        # --num_bit 2 --freeze_model \

        done
    done
done
