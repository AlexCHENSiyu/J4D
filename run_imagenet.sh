# bash ./run_imagenet.sh

# quantization based initialization
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=10427 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                    --dataset "imagenet" --model "mobilenet_v2" --num_workers 64 --data_path "/home/l44ye/datasets/" \
                                    --epochs 20 --trial 1 --batch_size 64 \
                                    --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                    --Q_initial_QF 10 --weight_lum 0.1 --weight_chrom 0.1 

# quantization based initialization
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3.9 -m torch.distributed.run --master_port=10427 -nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
#                                     --dataset "imagenet" --model "Swin_T" --num_workers 64 --data_path "/home/l44ye/datasets/" \
#                                     --epochs 20 --trial 1 --batch_size 64 \
#                                     --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
#                                     --Q_initial_Sensitivity 15 --weight_lum 0.01 --weight_chrom 0.01 \
#                                     --val_resize_size 232