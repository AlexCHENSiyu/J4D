GPU_ID=7

model=vgg13

student_list=(  
				"vgg8" \
				# "wrn_40_1"\
				# "ShuffleV1" \
				)

model_path=./save/models/${model}_vanilla/ckpt_epoch_240.pth
# [-r] for CE // [-a] for KL // [-b] for Loss_distill

Temp=4
kl_coeff=0
ce_coeff=$(python -c "print(1.0-$kl_coeff)")
distill_coeff_1=0
ratio=1
trial=1

for student in ${student_list[*]}; do
    for trial in 2; do # 2 3
        for distill_coeff_1 in 0.8; do # 2 3
            distill_coeff_2=$(python -c "print(${ratio}*${distill_coeff_1})")
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8  train_student.py --path_t ${model_path} \
                                        --num_workers 8  \
                                        --distill cov_kd -r ${ce_coeff} -a ${kl_coeff} -b ${distill_coeff_1} -l ${distill_coeff_2} \
                                        --trial ${trial} --kd_T ${Temp} \
                                        --model_s ${student} 

        done
    done
done