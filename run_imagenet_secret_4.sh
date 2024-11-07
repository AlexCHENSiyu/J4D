# conda activate CHEN
# bash ./run_imagenet.sh




weight_lum=10

for Q_initial_QF in 80 90; do
    CUDA_VISIBLE_DEVICES=6,7 python3.8 -m torch.distributed.run --master_port=25673 --nnodes=1 --nproc_per_node=2 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --QF_Inital \
                                        --dataset "imagenet" --model "mobilenet_v2" --num_workers 24 --data_path "/home/ahamsala/datasets/imagenet" \
                                        --epochs 20 --trial 1 --batch_size 128 \
                                        --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                        --Q_initial_QF ${Q_initial_QF} --weight_lum ${weight_lum} --weight_chrom ${weight_lum} --val_resize_size 232
done



