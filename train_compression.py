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
from torch.utils.data import DataLoader 
from models import load_model
from dataset.general_dataloader import load_dataset
from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train, validate , validate_jpeg
from helper.JPEG_layer import *
from dataset.cub2011 import *
import numpy as np
import random
import torch.multiprocessing as mp


def create_trial_folder(base_folder):
    trial_num = 1
    while True:
        trial_folder = os.path.join(base_folder, f"trial_{trial_num}")
        if not os.path.exists(trial_folder):
            os.makedirs(trial_folder)
            # print(f"Created folder: {trial_folder}")
            return trial_folder
        trial_num += 1


# python train_compression.py --JPEG_enable --freeze_model
# /usr/local/bin/python3.9 train_compression.py --JPEG_enable --freeze_model
# tmux
# tmux detach
# tmux attach -t 0
# tmux kill-session -t 0
# tmux ls
# tmux switch -t <session-name>
# tmux rename-session -t 0 <new-name>
# ctrl+b and press d to detach when running code
# ps aux | grep tmux
# salloc --time=0:30:0 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --account=def-ehyangit --mem=8G


def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    
    # optimization
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.4, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='110', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'densenet121',])
    parser.add_argument('--dataset', type=str, default='CUB200', choices=['CUB200'], help='dataset')
    parser.add_argument('--root', type=str, default='./data/cub2011', help='path to dataset')
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    
    # JPEG
    parser.add_argument('--JPEG_enable', action='store_true')
    parser.add_argument('--freeze_model', action='store_true')
    parser.add_argument('--fixed_alpha', action='store_true')
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')
    parser.add_argument('--QF_Inital', action='store_true')
    
    # JPEG Hyper
    parser.add_argument('--JPEG_alpha', type=float, default=20.0, help='Tempurature scaling')
    parser.add_argument('--hardness', type=float, default=3.0, help='Tempurature scaling')
    parser.add_argument('--JPEG_learning_rate', type=float, default=0.025, help='Quantization Table Learning Rate')
    parser.add_argument('--alpha_learning_rate', type=float, default=None, help='Alpha Learning Rate')
    parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=11, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=1.0, help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=255, help='Maximum Quantization Step')
    parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2,2**10 - 1), help='Window size for the reconstruction space')
    parser.add_argument('--weight_lum', type=float, default=0.0, help='weight for lum entropy')
    parser.add_argument('--weight_chrom', type=float, default=0.0, help='weight for Chrom entropy')
    
    parser.add_argument('--log_file', type=str, default='_alpha_untrainable', help='add text to the file', choices=['', '_alpha_per_q', '_alpha_per_channel', '_alpha_untrainable'])
    parser.add_argument('--seed', type=int, default=42, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of micro-batches to accumulate')
    
    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    
    if opt.JPEG_enable:
        opt.added_layer = "JPEG"
        if opt.fixed_alpha:
            opt.model_name = 'JPEG_lr_{}_alpha_{}_weight_{}_{}'.format(opt.JPEG_learning_rate, opt.JPEG_alpha, opt.weight_lum, opt.weight_chrom)
        else:
            opt.model_name = 'JPEG_lr_{}_weight_{}_{}_hardness_{}'.format(opt.JPEG_learning_rate, opt.weight_lum, opt.weight_chrom, opt.hardness)
    else:
        opt.added_layer = "vanilla"
        opt.model_name = '{}'.format(opt.model)
    
    base_path = './save/{}/{}'.format(opt.dataset, opt.model)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    base_folder = os.path.join(base_path, opt.model_name)
    opt.trial_folder = create_trial_folder(base_folder)

    # 1     2        3        4         5         6          7   8           9             
    # epoch best_acc test_acc train_acc test_loss train_loss bpp test_lum_dc test_chrom_dc 
    # 10          11            12           13             14           15             16    17   
    # test_lum_ac test_chrom_ac train_lum_dc train_chrom_dc train_lum_ac train_chrom_ac q_min q_max
    opt.logs_fname = os.path.join(opt.trial_folder, "logs.txt")
    with open(opt.logs_fname, 'a') as log:
        line =  'ep' + "\t" + \
                'best_acc' + "\t\t\t" +  \
                'test_acc' + "\t\t\t" + 'train_acc' + "\t\t\t" +  \
                'test_loss' + "\t\t\t" + 'train_loss' + "\t\t\t" + \
                'bpp' + "\t\t\t\t\t" + \
                'test_lum_dc' + "\t\t\t" + 'test_chrom_dc' + "\t\t" + \
                'test_lum_ac' + "\t\t\t" + 'test_chrom_ac' + "\t\t" + \
                'train_lum_dc' + "\t\t" + 'train_chrom_dc' + "\t\t" + \
                'train_lum_ac' + "\t\t" + 'train_chrom_ac' + "\t\t" + \
                'q_min' + "\t\t\t\t" + 'q_max' + '\t\t' + \
                'alpha_min' + "\t\t\t" + 'alpha_max' + '\n'
        log.write(line)
    
    opt.q_tables_folder = os.path.join(opt.trial_folder, "Q_tables")
    if not os.path.isdir(opt.q_tables_folder):
        os.makedirs(opt.q_tables_folder)
    
    opt.alpha_folder = os.path.join(opt.trial_folder, "Alpha_tables")
    if not os.path.isdir(opt.alpha_folder):
        os.makedirs(opt.alpha_folder)
    
    opt.opt_fname = os.path.join(opt.trial_folder, "opt.json")
    with open(opt.opt_fname, 'w') as f:
        json.dump(vars(opt), f, indent=4)

    return opt


def main(opt):
    best_acc = 0
    best_bpp = 0
    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # torch.autograd.set_detect_anomaly(True)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    mean_datatloader = (0, 0, 0)
    std_datatloader = (1/255., 1/255., 1/255.)

    if opt.dataset == 'CUB200':
        train_loader, val_loader = load_dataset(name="CUB200", batch_size=opt.batch_size, mean=mean_datatloader, std=std_datatloader, root=opt.root)
        n_cls = len(train_loader.dataset.class_names)
    else:
        raise NotImplementedError(opt.dataset)

    # model
    underlying_model = load_model(opt.model, num_classes=n_cls, pretrained=False)
    model_path = "./save/CUB200/pretrained/resnet18/resnet18_best.pth" 
    underlying_model.load_state_dict(torch.load(model_path)['model'])

    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=(224, 224, 3), mean=mean, std=std)        
        model = CustomModel(jpeg_layer, underlying_model)
        
        # for SGD optimizer
        # optimizer_data = [
        #     {'params': model.jpeg_layer.lum_qtable, 'lr': opt.JPEG_learning_rate, 'momentum': opt.momentum,},
        #     {'params': model.jpeg_layer.chrom_qtable, 'lr': (opt.JPEG_learning_rate), 'momentum': opt.momentum,},
        # ]

        # for Adam optimizer
        optimizer_data = [
            {'params': model.jpeg_layer.lum_qtable, },
            {'params': model.jpeg_layer.chrom_qtable, },
        ]
      
        if opt.JPEG_alpha_trainable:
            optimizer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': opt.alpha_learning_rate, 'momentum': opt.momentum,})
            optimizer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': (opt.alpha_learning_rate)/np.sqrt(2), 'momentum': opt.momentum,})
    else:
        model = underlying_model
        optimizer_data = [
            {'params': model.parameters(), 'lr': opt.learning_rate, 'momentum': opt.momentum, 'weight_decay': opt.weight_decay},
        ]

    optimizer = optim.Adam(optimizer_data, lr=1e-2, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(optimizer_data)
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        if not opt.seed:
            cudnn.benchmark = True
    
    # HDQ_preprocess = resize_compression(HDQ_transforms(QF_Y=10, QF_C=10))
    HDQ_preprocess = resize_compression(JPEG_layer_transforms(opt, img_shape=(224, 224, 3), mean=mean, std=std))
    val_loader.dataset.transform = HDQ_preprocess
    
    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        
        # train
        print("\n==> training...")
        time1 = time.time()
        train_acc, train_loss, train_lum_dc, train_chrom_dc, train_lum_ac, train_chrom_ac = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('Epoch [{}]: total time {:.2f} s'.format(epoch, time2 - time1))
        
        # catch the q_table and alpha for validate
        val_loader.dataset.transform.compressor.jpeg_layer.lum_qtable = model.jpeg_layer.lum_qtable.squeeze(0).detach().clone().cpu()
        val_loader.dataset.transform.compressor.jpeg_layer.chrom_qtable = model.jpeg_layer.chrom_qtable.squeeze(0).detach().clone().cpu()
        val_loader.dataset.transform.compressor.jpeg_layer.alpha_lum = model.jpeg_layer.alpha_lum.squeeze(0).detach().clone().cpu()
        val_loader.dataset.transform.compressor.jpeg_layer.alpha_chrom = model.jpeg_layer.alpha_chrom.squeeze(0).detach().clone().cpu()
        
        # validate
        print("\n==> validating...")
        time1 = time.time()
        test_acc, test_acc_top5, test_loss, bpp, test_lum_dc, test_chrom_dc, test_lum_ac, test_chrom_ac = validate_jpeg(val_loader, model.underlying_model, criterion, opt)
        time2 = time.time()
        print('Epoch [{}]: total time {:.2f} s'.format(epoch, time2 - time1))
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_bpp = bpp
        print('Best Accuracy: {:.2f} % \t BPP: {:.5f} '.format(best_acc, best_bpp))
        
        # save q_table
        if opt.JPEG_enable:
            lum_qtable =  model.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            chrom_qtable =  model.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0).cpu()
            print("Quantization Table --> Min: {:.5f}, Max: {:.5f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
            str_q_alpha = str(quantizationTable.min().item())  + "\t" + str(quantizationTable.max().item())
            
            # save the exact q_table
            q_tables_file = os.path.join(opt.q_tables_folder, 'q_table_epoch_{}.pt'.format(epoch))
            torch.save(quantizationTable, q_tables_file)
            
            # save lum and chrom q_table for ploting trend
            lum_qtable = lum_qtable.reshape(8,8)
            chrom_qtable = chrom_qtable.reshape(8,8)
            
            lum_q_tables_file = os.path.join(opt.q_tables_folder, 'lum_q_table.txt')
            with open(lum_q_tables_file, 'a') as f:
                np.savetxt(f, lum_qtable.cpu().numpy(), delimiter=',', fmt='%.04f')
                f.write('\n')
            chrom_q_tables_file = os.path.join(opt.q_tables_folder, 'chrom_q_table.txt')
            with open(chrom_q_tables_file, 'a') as f:
                np.savetxt(f, chrom_qtable.cpu().numpy(), delimiter=',', fmt='%.04f')
                f.write('\n')

        # save alpha_table
        if opt.JPEG_enable and (opt.JPEG_alpha_trainable or not opt.fixed_alpha):
            alpha_lum =  model.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha_chrom =  model.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alphaTable = torch.cat((alpha_lum, alpha_chrom), 0).cpu()
            print("Alpha --> Min: {:.5f}, Max: {:.5f}".format(alphaTable.min().item(), alphaTable.max().item()))
            str_q_alpha += "\t" +  str(alphaTable.min().item())  + "\t" + str(alphaTable.max().item())  + "\t" 
        
            # save the exact alpha_table
            alpha_tables_file = os.path.join(opt.alpha_folder, 'alpha_table_epoch_{}.pt'.format(epoch))
            torch.save(alphaTable, alpha_tables_file)
            
            # save lum and chrom alpha_table for ploting trend
            alpha_lum = alpha_lum.reshape(8,8)
            alpha_chrom = alpha_chrom.reshape(8,8)
            
            lum_alpha_tables_file = os.path.join(opt.alpha_folder, 'lum_alpha_table.txt')
            with open(lum_alpha_tables_file, 'a') as f:
                np.savetxt(f, alpha_lum.cpu().numpy(), delimiter=',', fmt='%.04f')
                f.write('\n')
            chrom_alpha_tables_file = os.path.join(opt.alpha_folder, 'chrom_alpha_table.txt')
            with open(chrom_alpha_tables_file, 'a') as f:
                np.savetxt(f, alpha_chrom.cpu().numpy(), delimiter=',', fmt='%.04f')
                f.write('\n')
        
        # save test train acc and loss
        with open(opt.logs_fname, 'a') as log:
            line =  f"{epoch}\t" + \
                    f"{best_acc.item():.15f}\t" +  \
                    f"{test_acc.cpu().numpy():.15f}\t" + f"{train_acc.cpu().numpy():.15f}\t" +  \
                    f"{test_loss:.15f}\t" + f"{train_loss:.15f}\t" + \
                    f"{bpp:.15f}\t" + \
                    f"{test_lum_dc:.15f}\t" + f"{test_chrom_dc:.15f}\t" + \
                    f"{test_lum_ac:.15f}\t" + f"{test_chrom_ac:.15f}\t" + \
                    f"{train_lum_dc:.15f}\t" + f"{train_chrom_dc:.15f}\t" + \
                    f"{train_lum_ac:.15f}\t" + f"{train_chrom_ac:.15f}\t" + \
                    str_q_alpha + '\n'
            log.write(line)

    # save model of last epoch
    state = { 'model': model.state_dict(),}
    save_file = os.path.join(opt.trial_folder, 'last.pth')
    torch.save(state, save_file)
    
    # After training, save the final q_table and alpha
    print("\n==> Finish training: ", opt.model_name)
    print('Best accuracy:', best_acc.item())


def check_val(opt):
    opt = parse_option()
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if opt.dataset == 'CUB200':
        train_loader, val_loader = load_dataset(name="CUB200", mean=mean, std=std)
        n_cls = len(train_loader.dataset.class_names)
    else:
        raise NotImplementedError(opt.dataset)
    
    # model
    model = load_model(opt.model, num_classes=n_cls, pretrained=False)
    
    # validate teacher accuracy
    model_path = "./save/CUB200/pretrained/resnet18/resnet18_best.pth" 
    model.load_state_dict(torch.load(model_path)['model'])
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    teacher_acc, _, _ = validate(val_loader, model, criterion, opt)
    
    print('Model accuracy: ', teacher_acc)


def profiling():
    import cProfile
    import pstats
 
    with cProfile.Profile() as pr:
        main()
 
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='profile_vanilla.prof')


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    timeStart = time.time()
    
    opt = parse_option()
    if opt.JPEG_enable:
        main(opt)
    else:
        check_val(opt)
        
    timeEnd = time.time()
    print('\nTotal time {:.2f} s'.format(timeEnd - timeStart))
    
    # profiling()
    
    # import torch
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))