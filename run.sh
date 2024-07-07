GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
                                    --JPEG_alpha 10 --hardness 3.0 --JPEG_learning_rate 0.025 \
                                    --weight_lum 0.005 --weight_chrom 0.005 \
                                    --num_workers 16

# JPEG_learning_rate_list=(0.015 0.020 0.025 0.030 0.035 0.040)
# for learning_rate in "${JPEG_learning_rate_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
#                                     --JPEG_alpha 20 --JPEG_learning_rate $learning_rate \
#                                     --weight_lum 0.1 --weight_chrom 0.1
# done

# weight_list=(0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17)
# for weight in "${weight_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
#                                     --JPEG_alpha 20 --JPEG_learning_rate 0.01 \
#                                     --weight_lum $weight --weight_chrom $weight
# done

# alpha_lr_pairs=(
#     "10 0.035"
#     "20 0.03"
#     "50 0.02"
#     "100 0.01"
#     "200 0.008"
# )

# alpha_lr_pairs=(
#     "200 0.008"
#     "300 0.007"
#     "400 0.006"
#     "500 0.005"
#     "600 0.004"
# )

# for pair in "${alpha_lr_pairs[@]}"; do
#     read alpha learning_rate <<< "$pair"
#     alpha=$(echo "$alpha" | bc) # Ensures alpha is treated as a number
#     learning_rate=$(echo "$learning_rate" | bc) # Ensures learning_rate is treated as a number

#     CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
#                                     --JPEG_alpha "$alpha" --JPEG_learning_rate "$learning_rate" \
#                                     --weight_lum 0.1 --weight_chrom 0.1
# done