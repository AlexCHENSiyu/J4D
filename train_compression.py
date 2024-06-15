from __future__ import print_function

import os
import argparse
import socket
import time

# import tensorboard_logger.tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import load_model
from dataset.general_dataloader import load_dataset
from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train, validate , validate_jpeg
from helper.JPEG_layer import *

import numpy as np
import random

# python train_compression.py --JPEG_enable --freeze_model -t 1
def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    
    # optimization
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,150', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'densenet121',])
    parser.add_argument('--dataset', type=str, default='CUB200', choices=['CUB200'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # JPEG
    parser.add_argument('--JPEG_enable', action='store_true')
    parser.add_argument('--freeze_model', action='store_true')
    parser.add_argument('--JPEG_alpha_trainable', action='store_true')
    parser.add_argument('--JEPG_alpha', type=float, default=100.0, help='Tempurature scaling')
    parser.add_argument('--JEPG_learning_rate', type=float, default=0.0125, help='Quantization Table Learning Rate')
    parser.add_argument('--alpha_learning_rate', type=float, default=None, help='Alpha Learning Rate')
    parser.add_argument('--Q_inital', type=float, default=1.0, help='Initial Quantization Step')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=11, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=1.0, help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=255, help='Maximum Quantization Step')
    parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2,2**10 - 1), help='Window size for the reconstruction space')
    parser.add_argument('--weight_lum', type=float, default=1, help='weight for lum entropy')
    parser.add_argument('--weight_chrom', type=float, default=1, help='weight for Chrom entropy')

    parser.add_argument('--weight_lum_dc', type=float, default=0.00001, help='weight for lum entropy')
    parser.add_argument('--weight_chrom_dc', type=float, default=0.0001, help='weight for Chrom entropy')
    
    
    parser.add_argument('--log_file', type=str, default='_alpha_untrainable', help='add text to the file', choices=['', '_alpha_per_q', '_alpha_per_channel', '_alpha_untrainable'])
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of micro-batches to accumulate')
    
    opt = parser.parse_args()

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        # opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/{}/models'.format(opt.dataset)
        # opt.tb_path = './save/tensorboard'
        opt.log_pth = './save/{}/log/'.format(opt.dataset)
        opt.Q_tables_pth = './save/{}/Q_tables/'.format(opt.dataset)
        opt.alpha_pth = './save/{}/Alpha/'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.JPEG_enable:
        opt.added_layer = "JPEG"
        opt.model_name = '{}_{}_alpha_lr_{}_alpha_{}_JEPG_lr_{}{}_trial_{}'.format(opt.dataset, opt.model, opt.alpha_learning_rate, opt.JEPG_alpha, opt.JEPG_learning_rate, opt.log_file, opt.trial)
    else:
        opt.added_layer = "vanilla"
        opt.model_name = '{}_{}_trial_{}'.format(opt.dataset, opt.model,  opt.trial)

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_key = '{}_lr_{}_decay_{}_log_{}'.format(opt.model, opt.learning_rate, opt.weight_decay, opt.added_layer)

    opt.log_folder = os.path.join(opt.log_pth, opt.log_key)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    opt.alpha_folder = os.path.join(opt.alpha_pth, opt.log_key)
    if not os.path.isdir(opt.alpha_folder):
        os.makedirs(opt.alpha_folder)

    opt.Q_tables_folder = os.path.join(opt.Q_tables_pth, opt.log_key)
    if not os.path.isdir(opt.Q_tables_folder):
        os.makedirs(opt.Q_tables_folder)

    return opt


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


def main(opt):
    best_acc = 0
    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        cudnn.benchmark = True

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    mean_datatloader = (0, 0, 0)
    std_datatloader = (1/255., 1/255., 1/255.)

    if opt.dataset == 'CUB200':
        train_loader, val_loader = load_dataset(name="CUB200", mean=mean_datatloader, std=std_datatloader)
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
        
        optimizer_data = [
                            # {'params': model.underlying_model.parameters(), 'lr': opt.learning_rate, 'momentum': opt.momentum, 'weight_decay': opt.weight_decay},
                            {'params': model.jpeg_layer.lum_qtable, 'lr': opt.JEPG_learning_rate, 'momentum': opt.momentum,},
                            {'params': model.jpeg_layer.chrom_qtable, 'lr': (opt.JEPG_learning_rate), 'momentum': opt.momentum,},
                            # {'params': model.jpeg_layer.chrom_qtable, 'lr': (opt.JEPG_learning_rate)/np.sqrt(2), 'momentum': opt.momentum,},
                         ]
        
        if opt.JPEG_alpha_trainable:
            optimizer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': opt.alpha_learning_rate, 'momentum': opt.momentum,})
            optimizer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': (opt.alpha_learning_rate)/np.sqrt(2), 'momentum': opt.momentum,})
    else:
        model =  underlying_model
        optimizer_data = [
            {'params': model.parameters(), 'lr': opt.learning_rate, 'momentum': opt.momentum, 'weight_decay': opt.weight_decay},
        ]
    
    optimizer = optim.SGD(optimizer_data)
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        if not opt.seed:
            cudnn.benchmark = True
    
    # log file
    log_fname = os.path.join(opt.log_folder, '{experiment}.txt'.format(experiment=opt.model_name))

    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        if opt.JPEG_enable:
            lum_qtable =  model.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            chrom_qtable =  model.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0)
            print("Quantization Table --> Min: {:.2f}, Max: {:.2f}".format(quantizationTable.min().item(), quantizationTable.max().item()))
            str_q_alpha = str(quantizationTable.min().item())  + "\t" + str(quantizationTable.max().item())
            
        if opt.JPEG_enable and opt.JPEG_alpha_trainable:
            alpha_lum =  model.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha_chrom =  model.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
            alpha = torch.cat((alpha_lum, alpha_chrom), 0)
            print("Alpha --> Min: {:.5f}, Max: {:.5f}".format(alpha.min().item(), alpha.max().item()))
            str_q_alpha += "\t" +  str(alpha.min().item())  + "\t" + str(alpha.max().item())  + "\t" 
            
        # train and validate
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)    
        test_acc, test_acc_top5, test_loss = validate_jpeg(val_loader, model, criterion, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'best_acc': best_acc,
                'opt': opt,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        with open(log_fname, 'a') as log:
            line = str(epoch) + "\t" + \
                   str(test_acc.cpu().numpy()) + "\t" +  \
                   str(best_acc.item()) + "\t" +  \
                   str(test_loss)   + "\t" + \
                   str(train_loss)   + "\t" + \
                   str_q_alpha + '\n'
            log.write(line)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'opt': opt,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            # torch.save(state, save_file)
        
        # if epoch > 1: break

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc.item())
    QF_fname = os.path.join(opt.Q_tables_folder, '{experiment}.pt'.format(experiment=opt.model_name))
    alpha_fname = os.path.join(opt.alpha_folder, '{experiment}.pt'.format(experiment=opt.model_name))
    
    if opt.JPEG_enable:
        lum_qtable =  model.jpeg_layer.lum_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        chrom_qtable =  model.jpeg_layer.chrom_qtable.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        quantizationTable = torch.cat((lum_qtable, chrom_qtable), 0).cpu()
        torch.save(quantizationTable, QF_fname)
    
    if opt.JPEG_enable and opt.JPEG_alpha_trainable: 
        alpha_lum =  model.jpeg_layer.alpha_lum.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        alpha_chrom =  model.jpeg_layer.alpha_chrom.squeeze(0).squeeze(1).squeeze(-1).clone().detach()
        alpha = torch.cat((alpha_lum, alpha_chrom), 0).cpu()
        torch.save(alpha, alpha_fname)
    
    exp_txt = open('{}_experimetal_Results_JPEG_DL.txt'.format(opt.dataset), 'a+')
    exp_txt.write(opt.model_name +"\t"+ str(epoch) + "\t" + str(best_acc.item()) + "\n") # Write some text
    exp_txt.close() # Close the file

    # save model
    state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'best_acc': best_acc,
            'accuracy': test_acc,
            'opt': opt,
            'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


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
    opt = parse_option()
    if opt.JPEG_enable:
        main(opt)
    else:
        check_val(opt)
    # # profiling()

    
    
    # import torch
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))