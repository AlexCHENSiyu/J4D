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
from helper.loops import validate_jpeg
from helper.JPEG_layer import *

import numpy as np
import random


# base_path = "./save/CUB200/resnet18/JPEG_lr_0.005_alpha_100.0_weight_0.5_0.5/trial_1"
base_path ="./save/CUB200/resnet18/JPEG_lr_0.01_alpha_100.0_weight_0.1_0.1_QF_70/trial_1"

def main(opt):
    opt.seed = 0
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
        n_cls = len(train_loader.dataset.class_names) # 200
    else:
        raise NotImplementedError(opt.dataset)

    for QF_value in range(1, 101):
        # load underlying model
        underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
        model_path = "./save/CUB200/pretrained/resnet18/resnet18_best.pth"
        underlying_model.load_state_dict(torch.load(model_path)['model'])
        
        # load q_table stored
        # JPEG_layer_q_table = torch.load(os.path.join(base_path,'Q_tables/q_table_epoch_10.pt'))
        # lum_q_table, chrom_q_table = JPEG_layer_q_table[0], JPEG_layer_q_table[1]
        # lum_q_table = lum_q_table.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # chrom_q_table = chrom_q_table.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        
        # lum_q_table = torch.tensor(quantizationTable(50,True), dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # chrom_q_table = torch.tensor(quantizationTable(50,False), dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        
        # if opt.JPEG_enable:
        #     jpeg_layer = JPEG_layer(opt=opt, img_shape=(224, 224, 3), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        #     jpeg_layer.lum_qtable = nn.Parameter(lum_q_table)
        #     jpeg_layer.chrom_qtable = nn.Parameter(chrom_q_table)
        #     model = CustomModel(jpeg_layer, underlying_model)
        
        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            underlying_model.cuda()
            # model = model.cuda()
            criterion = criterion.cuda()
            if not opt.seed:
                cudnn.benchmark = True
        
        HDQ_preprocess = resize_compression(HDQ_transforms(QF_Y = QF_value, QF_C = QF_value))
        # HDQ_preprocess = resize_compression(JPEG_layer_transforms(opt, img_shape=(224, 224, 3), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        val_loader.dataset.transform = HDQ_preprocess
        
        # val_loader.dataset.transform.compressor.jpeg_layer.lum_qtable = model.jpeg_layer.lum_qtable.squeeze(0).detach().clone().cpu()
        # val_loader.dataset.transform.compressor.jpeg_layer.chrom_qtable = model.jpeg_layer.chrom_qtable.squeeze(0).detach().clone().cpu()
        # val_loader.dataset.transform.compressor.jpeg_layer.alpha_lum = model.jpeg_layer.alpha_lum.squeeze(0).detach().clone().cpu()
        # val_loader.dataset.transform.compressor.jpeg_layer.alpha_chrom = model.jpeg_layer.alpha_chrom.squeeze(0).detach().clone().cpu()
        
        test_acc, test_acc_top5, test_loss, bpp, lum_dc, chrom_dc, lum_ac, chrom_ac = validate_jpeg(val_loader, underlying_model, criterion, opt)
        
        print("test_acc:", test_acc.item())
        # print("test_acc_top5:", test_acc_top5.item())
        print("test_loss:", test_loss)
        print("bpp:", bpp)
        print("lum_dc:", lum_dc)
        print("chrom_dc:", chrom_dc)
        print("lum_ac:", lum_ac)
        print("chrom_ac:", chrom_ac)
    
        with open("./save/CUB200/resnet18/original_HDQ.txt", 'a') as log:
            line =  f"{QF_value}\t" + \
                    f"{test_acc.cpu().numpy():.15f}\t" + \
                    f"{bpp:.15f}\t" + "\n"
            log.write(line)

if __name__ == '__main__':
    opt_json_filepath = os.path.join(base_path,'opt.json')
    with open(opt_json_filepath, 'r') as f:
        data = json.load(f)
    opt = argparse.Namespace(**data)
    # print(opt.JPEG_alpha)
    
    if opt.JPEG_enable:
        main(opt)