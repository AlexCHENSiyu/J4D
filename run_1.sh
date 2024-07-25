GPU_ID=0

# CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model --QF_Inital --fixed_alpha \
#                                     --JPEG_alpha 10 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 5.0 --weight_chrom 5.0 --epochs 10 \
#                                     --num_workers 16


# CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model --fixed_alpha \
#                                     --JPEG_alpha 100 --JPEG_learning_rate 0.1 \
#                                     --weight_lum 1.0 --weight_chrom 1.0 \
#                                     --num_workers 16


# CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --JPEG_alpha 100 --JPEG_learning_rate 0.01 --Q_initial_QF 40 \
#                                     --weight_lum 0.001 --weight_chrom 0.001 \
#                                     --num_workers 16 --optimizer "sgd"

# JPEG_learning_rate_list=(0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0)
# for learning_rate in "${JPEG_learning_rate_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --JPEG_alpha 20 --hardness 20.0 --JPEG_learning_rate $learning_rate \
#                                     --weight_lum 2.0 --weight_chrom 2.0 \
#                                     --num_workers 16
# done

# weight_list=(0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0)
# for weight in "${weight_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --JPEG_alpha 10 --hardness 20.0 --JPEG_learning_rate 0.05 \
#                                     --weight_lum $weight --weight_chrom $weight \
#                                     --num_workers 16
# done

QF_list=(10 20 30 40 50 60 70 80 90 100)
for QF in "${QF_list[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} ~/miniconda3/envs/CHEN/bin/python3.9 train_compression.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --JPEG_alpha 100 --JPEG_learning_rate 0.01 --Q_initial_QF $QF \
                                    --weight_lum 0.0 --weight_chrom 0.0 \
                                    --num_workers 20
done

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