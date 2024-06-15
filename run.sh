GPU_ID=7


teacher_list=(  
                # CUB 200
                "resnet18"
				)


JEPG_learning_rate=0.01

for teacher in ${teacher_list[*]}; do
    for trial in 1; do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_compression.py --model ${teacher} -t ${trial} \
                                            --JPEG_enable --JEPG_learning_rate ${JEPG_learning_rate} \
                                            --num_bit 11 --num_non_zero_q 5 --weight_lum 0.0005 --weight_chrom 0.0005 --JEPG_alpha 1 \
                                            --weight_lum_dc 0.1 --weight_lum_dc 0.1 \
                                            --JEPG_learning_rate 0.2
    done
done