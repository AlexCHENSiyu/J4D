GPU_ID=0

# CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
#                                     --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
#                                     --JEPG_alpha 5 --JEPG_learning_rate 0.038 \
#                                     --weight_lum 0.0 --weight_chrom 0.0 \

JEPG_learning_rate_list=(0.015 0.020 0.025 0.030 0.035 0.040)

for learning_rate in "${JEPG_learning_rate_list[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
                                    --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
                                    --JEPG_alpha 20 --JEPG_learning_rate $learning_rate \
                                    --weight_lum 0.0 --weight_chrom 0.0
done