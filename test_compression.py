from __future__ import print_function

import os
import argparse
import socket
import time
import json

# import tensorboard_logger.tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import load_model
from dataset.general_dataloader import load_dataset
from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train, validate, validate_with_output
from helper.JPEG_layer import *

import numpy as np
import random


base_path = "./save/CUB200/resnet18/JPEG_lr_0.01_alpha_100.0_alpha_lr_None_weight_0.1_0.1_alpha_untrainable/trial_2"


def main(opt):
    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("set seed:", opt.seed)

    if opt.dataset == 'CUB200':
        train_loader, val_loader = load_dataset(name="CUB200", mean=(0, 0, 0), std=(1/255., 1/255., 1/255.))
        targets = []
        for inputs, batch_targets in val_loader:
            targets.extend(batch_targets)
            
        n_cls = len(train_loader.dataset.class_names) # 200
    else:
        raise NotImplementedError(opt.dataset)

    # model
    underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
    model_path = "./save/CUB200/pretrained/resnet18/resnet18_best.pth"
    underlying_model.load_state_dict(torch.load(model_path)['model'])
    
    JPEG_layer_q_table = torch.load(os.path.join(base_path,'Q_tables/q_table_epoch_100.pt'))
    lum_q_table, chrom_q_table = JPEG_layer_q_table[0], JPEG_layer_q_table[1]
    lum_q_table = lum_q_table.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    chrom_q_table = chrom_q_table.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    
    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=(224, 224, 3), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        jpeg_layer.lum_qtable = nn.Parameter(lum_q_table)
        jpeg_layer.chrom_qtable = nn.Parameter(chrom_q_table)
        model = CustomModel(jpeg_layer, underlying_model)
    
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        if not opt.seed:
            cudnn.benchmark = True
    
    test_acc, test_acc_top5, test_loss, outputs, lum_dc, chrom_dc, lum_ac, chrom_ac = validate_with_output(val_loader, model, criterion, opt)
    print("test_acc:", test_acc)
    print("test_acc_top5:", test_acc_top5)
    print("lum_dc:", lum_dc)
    print("chrom_dc:", chrom_dc)
    print("lum_ac:", lum_ac)
    print("chrom_ac:", chrom_ac)
    print("test_loss:", test_loss)
    
    with open("./save/CUB200/resnet18/output_target.txt", 'w') as f:
        for output, target in zip(outputs, targets):
            f.write(f"{output} {target}\n")


if __name__ == '__main__':
    opt_json_filepath = os.path.join(base_path,'opt.json')
    with open(opt_json_filepath, 'r') as f:
        data = json.load(f)
    opt = argparse.Namespace(**data)
    # print(opt.JPEG_alpha)
    
    if opt.JPEG_enable:
        main(opt)