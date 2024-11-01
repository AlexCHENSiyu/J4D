# conda activate CHEN
# bash ./run_imagenet.sh

# This is for QF training
# QF_WEIGHT = [(10,50),(10,20),(12,20),(14,20),(16,20),(18,20),(20,20),(16,10),(18,10),(20,10),(30,10),(40,10),(50,10),(60,10)]
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=25671 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "Swin_T" --num_workers 64 --data_path "/home/l44ye/datasets/" \
#                                     --epochs 20 --trial 1 --batch_size 64 \
#                                     --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                     --Q_initial_QF 70 --weight_lum 10.0 --weight_chrom 10.0 \
#                                     --val_resize_size 232 

# This is for sensitivity training
# weight = [10.0, 5.0, 1.0, 0.1, 0.01]

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=25671 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
                                    --dataset "imagenet" --model "ConvNeXt_tiny" --num_workers 64 --data_path "/home/l44ye/datasets/" \
                                    --epochs 20 --trial 1 --batch_size 64 \
                                    --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                    --Q_initial_Sensitivity 10 --weight_lum 0.01 --weight_chrom 0.01 \
                                    --val_resize_size 236

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=25671 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
                                    --dataset "imagenet" --model "ConvNeXt_tiny" --num_workers 64 --data_path "/home/l44ye/datasets/" \
                                    --epochs 20 --trial 1 --batch_size 64 \
                                    --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                    --Q_initial_Sensitivity 5 --weight_lum 0.01 --weight_chrom 0.01 \
                                    --val_resize_size 236


# for weight_lum in 0.02 0.05 0.2 0.5 2 5; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python3.9 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
#                                         --dataset "imagenet" --model "ConvNeXt_tiny" --num_workers 64 --data_path "/home/l44ye/datasets/" \
#                                         --epochs 20 --trial 1 --batch_size 64 \
#                                         --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                         --Q_initial_Sensitivity 50 --weight_lum ${weight_lum} --weight_chrom ${weight_lum} \
#                                         --val_resize_size 236
# done


# This is an example how to resume training(train more epochs)
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3.9 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "mobilenet_v2" --num_workers 128 --data_path "/home/l44ye/datasets/" \
#                                     --epochs 20 --trial 1 --batch_size 64 \
#                                     --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                     --Q_initial_QF 10 --weight_lum 50.0 --weight_chrom 50.0 --resume "./save/imagenet/mobilenet_v2/JPEG_lr_0.01_alpha_100.0_weight_50.0_50.0_QF_10/trial_1" \


# get HDQ_baseline
# CUDA_VISIBLE_DEVICES=0,1 python3.9 -m torch.distributed.run --master_port=25671 --nnodes=1 --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "Swin_T" --num_workers 32 --data_path "/home/l44ye/datasets/" \
#                                     --epochs 20 --trial 1 --batch_size 64 \
#                                     --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                     --Q_initial_QF 10 --weight_lum 20.0 --weight_chrom 20.0 \
#                                     --val_resize_size 232 --HDQ_baseline
