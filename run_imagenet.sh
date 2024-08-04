# torchrun --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "resnet18" --epoch 11 \
#                                     --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
#                                     --weight_lum 0.1 --weight_chrom 0.1 --num_workers 16 \
#                                     --lr_step_size 1 --lr_gamma 0.98 \
#                                     --trial 1 --batch_size 64 --resume "./save/imagenet/resnet18/JPEG_lr_0.01_alpha_100.0_weight_0.1_0.1_QF_10/trial_1"


GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ~/miniconda3/envs/CHEN/bin/python3.9 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "mobilenet_v2" --epoch 1 \
                                    --JPEG_alpha 100 --Q_initial_QF 10 --JPEG_learning_rate 0.01 \
                                    --weight_lum 0.01 --weight_chrom 0.01 --num_workers 16 \
                                    --lr_step_size 1 --lr_gamma 0.98 \
                                    --trial 1 --batch_size 64 --HDQ_baseline