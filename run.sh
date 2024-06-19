GPU_ID=0

# CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
#                                     --JEPG_alpha 50 --JEPG_learning_rate 0.025 \
#                                     --weight_lum 0.0 --weight_chrom 0.0 \

JEPG_learning_rate_list=(0.015 0.020 0.025 0.030 0.035 0.040)
for learning_rate in "${JEPG_learning_rate_list[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
                                    --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
                                    --JEPG_alpha 20 --JEPG_learning_rate $learning_rate \
                                    --weight_lum 0.1 --weight_chrom 0.1
done

# weight_list=(0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17)
# for weight in "${weight_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
#                                     --JEPG_alpha 20 --JEPG_learning_rate 0.01 \
#                                     --weight_lum $weight --weight_chrom $weight
# done