# conda activate CHEN
# bash ./run_imagenet.sh


# QF_WEIGHT = [(10,50),(10,20),(12,20),(14,20),(16,20),(18,20),(20,20),(16,10),(18,10),(20,10),(30,10),(40,10),(50,10),(60,10)]
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3.8 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "Mnasnet" --num_workers 32 --data_path "/home/ahamsala/datasets/imagenet" \
#                                     --epochs 20 --trial 1 --batch_size 64 \
#                                     --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                     --Q_initial_QF 10 --weight_lum 50.0 --weight_chrom 50.0 \


# This is for sensitivity training
# weight = [10.0, 5.0, 1.0, 0.1, 0.01]
CUDA_VISIBLE_DEVICES=4,5,6,7 python3.8 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
                                    --dataset "imagenet" --model "Mnasnet" --num_workers 32 --data_path "/home/ahamsala/datasets/imagenet" \
                                    --epochs 20 --trial 1 --batch_size 64 \
                                    --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                    --Q_initial_Sensitivity 50 --weight_lum 10.0 --weight_chrom 10.0  \


# This is an example how to resume training(train more epochs)
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3.9 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
#                                     --dataset "imagenet" --model "mobilenet_v2" --num_workers 24 --data_path "/home/ahamsala/datasets/imagenet" \
#                                     --epochs 20 --trial 1 --batch_size 64 \
#                                     --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                     --Q_initial_QF 10 --weight_lum 50.0 --weight_chrom 50.0 --resume "./save/imagenet/mobilenet_v2/JPEG_lr_0.01_alpha_100.0_weight_50.0_50.0_QF_10/trial_1" \

