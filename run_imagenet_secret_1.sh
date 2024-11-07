weight_lum=0.02

# source ~/PROJECT_AH/CKD_venv/bin/activate
for Q_initial_QF in 5 10 15; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3.8 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
                                        --dataset "imagenet" --model "mobilenet_v2" --num_workers 24 --data_path "/home/ahamsala/datasets/imagenet" \
                                        --epochs 20 --trial 1 --batch_size 64 \
                                        --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                        --Q_initial_Sensitivity ${Q_initial_QF} --weight_lum ${weight_lum} --weight_chrom ${weight_lum}
done