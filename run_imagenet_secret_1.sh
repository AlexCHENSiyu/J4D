# conda activate CHEN
# bash ./run_imagenet.sh


Q_initial_Sensitivity=10


for weight_lum in 0.01 0.1 1 10; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3.8 -m torch.distributed.run --master_port=25670 --nnodes=1 --nproc_per_node=4 train_imagenet.py --JPEG_enable --freeze_model --fixed_alpha --Sensitivity_Inital \
                                        --dataset "imagenet" --model "Mnasnet" --num_workers 32 --data_path "/home/ahamsala/datasets/imagenet" \
                                        --epochs 20 --trial 1 --batch_size 64 \
                                        --JPEG_alpha 100  --JPEG_learning_rate 0.01 \
                                        --Q_initial_Sensitivity ${Q_initial_Sensitivity} --weight_lum ${weight_lum} --weight_chrom ${weight_lum}
done
