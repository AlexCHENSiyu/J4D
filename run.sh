GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model \
                                    --num_bit 11 --num_non_zero_q 5 --momentum 0.9 \
                                    --JEPG_alpha 10 --JEPG_learning_rate 0.030 \
                                    --weight_lum 0.0 --weight_chrom 0.0 \
