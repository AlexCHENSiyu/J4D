# torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "mobilenet_v3" \
#                                     --JPEG_alpha 100 --JPEG_learning_rate 0.01 --Q_initial_QF 10 --epoch 20 \
#                                     --weight_lum 20.0 --weight_chrom 20.0 --num_workers 24 \
#                                     --lr_step_size 2 --lr_gamma 0.973 \
#                                     --auto_augment imagenet --random_erase 0.2 \
#                                     --trial 1 --batch_size 32


# ~/miniconda3/envs/CHEN/bin/python3.9 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "Mnasnet" --epoch 1 \
#                                     --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 24 \
#                                     --lr_step_size 1 --lr_gamma 0.98 \
#                                     --trial 1 --batch_size 32 --test_only --HDQ_baseline


torchrun --nproc_per_node=1 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "Mnasnet" \
                                    --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
                                    --weight_lum 20.0 --weight_chrom 20.0 --num_workers 24 \
                                    --lr_step_size 1 --lr_gamma 0.98 \
                                    --trial 1 --batch_size 32 --HDQ_baseline


# torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "ConvNeXt_tiny" --epoch 15\
#                                     --JPEG_alpha 100 --Q_initial_QF 100 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 48 \
#                                     --trial 1 --batch_size 16 \
#                                     --learning_rate 1e-3 --lr_scheduler cosineannealinglr \
#                                     --lr_warmup_epochs 5 --lr_warmup_method linear --auto_augment ta_wide --random_erase 0.1 \
#                                     --label_smoothing 0.1 --mixup_alpha 0.2 --cutmix_alpha 1.0 --weight_decay 0.05 --norm_weight_decay 0.0 \
#                                     --train_crop_size 176 --model_ema --val_resize_size 236 --ra_sampler --ra_reps 4