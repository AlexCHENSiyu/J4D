# torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "resnet18" --epoch 10 \
#                                     --JPEG_alpha 100 --Q_initial_QF 40 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 32 \
#                                     --lr_step_size 1 --lr_gamma 0.98 \
#                                     --trial 1
GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} ~/miniconda3/envs/CHEN/bin/python3.9 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "resnet18" --epoch 10 \
                                    --JPEG_alpha 100 --Q_initial_QF 40 --JPEG_learning_rate 0.01 \
                                    --weight_lum 0.1 --weight_chrom 0.1 --num_workers 16 \
                                    --lr_step_size 1 --lr_gamma 0.98 \
                                    --trial 1 --test_onlyd