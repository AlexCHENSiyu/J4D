GPU_ID=0,1

CUDA_VISIBLE_DEVICES=${GPU_ID} ~/miniconda3/envs/test/bin/python3.9 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --lr-step-size 1 --lr-gamma 0.98 \
                                    --dataset "imagenet" --model "mobileNet_v2" --epoch 20 \
                                    --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                    --weight_lum 0.1 --weight_chrom 0.1 \
                                    --num_workers 16