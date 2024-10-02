from __future__ import print_function, division
import sys
import time
import torch
import torch.nn.functional as F
from .util import AverageMeter, accuracy
from .JPEG_layer import *
from torchvision import transforms
# from torchviz import make_dot


def train(epoch, train_loader, model, criterion, optimizer, opt, backward=True):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_lum_dc_entropy = AverageMeter()
    losses_chrom_dc_entropy = AverageMeter()
    losses_lum_ac_entropy = AverageMeter()
    losses_chrom_ac_entropy = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.to(opt.device)
            target = target.to(opt.device)

        # ===================forward=====================
        output, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy = model(input)
        loss = criterion(output, target) + opt.weight_lum * (lum_dc_entropy + lum_ac_entropy) + \
                opt.weight_chrom * (chrom_dc_entropy + chrom_ac_entropy)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # ===================backward=====================
        if backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # ===================meters=====================
        
        losses.update(loss.item(), input.size(0))
        losses_lum_dc_entropy.update(lum_dc_entropy.item(), input.size(0))
        losses_chrom_dc_entropy.update(chrom_dc_entropy.item(), input.size(0))
        losses_lum_ac_entropy.update(lum_ac_entropy.item(), input.size(0))
        losses_chrom_ac_entropy.update(chrom_ac_entropy.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.6f} ({batch_time.avg:.6f})\t'
                  'Data {data_time.val:.6f} ({data_time.avg:.6f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'Lum_DC {loss_lum_dc_entropy.val:.3f} ({loss_lum_dc_entropy.avg:.3f})\t'
                  'Chrom_DC {loss_chrom_dc_entropy.val:.3f} ({loss_chrom_dc_entropy.avg:.3f})\t'
                  'Lum_AC {loss_lum_ac_entropy.val:.3f} ({loss_lum_ac_entropy.avg:.3f})\t'
                  'Chrom_AC {loss_chrom_ac_entropy.val:.3f} ({loss_chrom_ac_entropy.avg:.3f})\t'
                  'Acc@1 {top1.val:.6f} ({top1.avg:.6f})\t'
                  'Acc@5 {top5.val:.6f} ({top5.avg:.6f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,data_time=data_time,
                   loss=losses, loss_lum_dc_entropy=losses_lum_dc_entropy, loss_chrom_dc_entropy=losses_chrom_dc_entropy,
                   loss_lum_ac_entropy = losses_lum_ac_entropy, loss_chrom_ac_entropy = losses_chrom_ac_entropy,
                   top1=top1, top5=top5))
            sys.stdout.flush()

        # with open(opt.train_log_fname, 'a') as log:
        #     line =  f"{top1.avg.item():.15f}\t" +  \
        #             f"{losses.avg:.15f}\t" + \
        #             f"{losses_lum_dc_entropy.avg:.15f}\t" + f"{losses_chrom_dc_entropy.avg:.15f}\t" + \
        #             f"{losses_lum_ac_entropy.avg:.15f}\t" + f"{losses_chrom_ac_entropy.avg:.15f}\t" + '\n'
        #     log.write(line)
            
    print(' * Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}'.format(top1=top1, top5=top5))
    
    return top1.avg, losses.avg, losses_lum_dc_entropy.avg, losses_chrom_dc_entropy.avg, losses_lum_ac_entropy.avg, losses_chrom_ac_entropy.avg


def validate_jpeg(val_loader, model, criterion, opt):
    """validation"""
    # switch to evaluate mode
    model.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_lum_dc_entropy = AverageMeter()
    losses_chrom_dc_entropy = AverageMeter()
    losses_lum_ac_entropy = AverageMeter()
    losses_chrom_ac_entropy = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    BPP = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            (input, bpp, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy), target = data
            
            input = input.float()
            if torch.cuda.is_available():
                input = input.to(opt.device)
                target = target.to(opt.device)
                lum_dc_entropy = lum_dc_entropy.to(opt.device).mean() 
                chrom_dc_entropy = chrom_dc_entropy.to(opt.device).mean() 
                lum_ac_entropy = lum_ac_entropy.to(opt.device).mean() 
                chrom_ac_entropy = chrom_ac_entropy.to(opt.device).mean() 

            # ===================inference=====================
            output = model(input)
            loss = criterion(output, target) + opt.weight_lum * (lum_dc_entropy + lum_ac_entropy) + \
                    opt.weight_chrom * (chrom_dc_entropy + chrom_ac_entropy)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            losses.update(loss.item(), input.size(0))
            losses_lum_dc_entropy.update(lum_dc_entropy.item(), input.size(0))
            losses_chrom_dc_entropy.update(chrom_dc_entropy.item(), input.size(0))
            losses_lum_ac_entropy.update(lum_ac_entropy.item(), input.size(0))
            losses_chrom_ac_entropy.update(chrom_ac_entropy.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            BPP.update(bpp.mean().item(), input.size(0))

            # ===================meters=====================
            
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(  'Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.6f} ({batch_time.avg:.6f})\t'
                        'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                        'BPP {BPP.val:.6f} ({BPP.avg:.6f})\t'
                        'Lum_DC {loss_lum_dc_entropy.val:.3f} ({loss_lum_dc_entropy.avg:.3f})\t'
                        'Chrom_DC {loss_chrom_dc_entropy.val:.3f} ({loss_chrom_dc_entropy.avg:.3f})\t'
                        'Lum_AC {loss_lum_ac_entropy.val:.3f} ({loss_lum_ac_entropy.avg:.3f})\t'
                        'Chrom_AC {loss_chrom_ac_entropy.val:.3f} ({loss_chrom_ac_entropy.avg:.3f})\t'
                        'Acc@1 {top1.val:.6f} ({top1.avg:.6f})\t'
                        'Acc@5 {top5.val:.6f} ({top5.avg:.6f})'.format(
                        idx, len(val_loader), batch_time=batch_time, 
                        loss=losses, BPP=BPP, loss_lum_dc_entropy=losses_lum_dc_entropy, loss_chrom_dc_entropy=losses_chrom_dc_entropy,
                        loss_lum_ac_entropy = losses_lum_ac_entropy, loss_chrom_ac_entropy = losses_chrom_ac_entropy,
                        top1=top1, top5=top5))

            # with open(opt.val_log_fname, 'a') as log:
            #     line =  f"{top1.avg.item():.15f}\t" +  \
            #             f"{losses.avg:.15f}\t" + \
            #             f"{losses_lum_dc_entropy.avg:.15f}\t" + f"{losses_chrom_dc_entropy.avg:.15f}\t" + \
            #             f"{losses_lum_ac_entropy.avg:.15f}\t" + f"{losses_chrom_ac_entropy.avg:.15f}\t" + '\n'
            #     log.write(line)

    print(' * Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg, BPP.avg, losses_lum_dc_entropy.avg, losses_chrom_dc_entropy.avg, losses_lum_ac_entropy.avg, losses_chrom_ac_entropy.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    entropy_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    BPP = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            (input, bpp), target = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.to(opt.device)
                target = target.to(opt.device)

            # compute output
            # breakpoint()
            output = model(input)

            loss = criterion(output, target) 
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            BPP.update(bpp.mean(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Entropy Loss {entropy_loss.val:.4f} ({entropy_loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'BPP {BPP.val:.3f} ({BPP.avg:.3f})'
                      .format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses, entropy_loss=entropy_losses, 
                       top1=top1, top5=top5, BPP=BPP))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg, BPP.avg


def validate_attack(val_loader, model, criterion, opt, atk):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()


    end = time.time()
    for idx, (input, target) in enumerate(val_loader):

        adv_images = atk(input, target)

        input = adv_images.float()
        if torch.cuda.is_available():
            input = input.to(opt.device)
            target = target.to(opt.device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_with_output(val_loader, model, criterion, opt):
    """validation"""
    # switch to evaluate mode
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_lum_dc_entropy = AverageMeter()
    losses_chrom_dc_entropy = AverageMeter()
    losses_lum_ac_entropy = AverageMeter()
    losses_chrom_ac_entropy = AverageMeter()

    with torch.no_grad():
        outputs = []
        
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.to(opt.device)
                targets = targets.to(opt.device)

            # compute output
            output, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy = model(inputs)
            outputs.extend(torch.argmax(output, dim=1).cpu().tolist()) # type: <class 'torch.Tensor'>, shape: torch.Size([16, 200])
            
            loss = criterion(output, targets) + opt.weight_lum * (lum_dc_entropy + lum_ac_entropy) + \
                opt.weight_chrom * (chrom_dc_entropy + chrom_ac_entropy)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            
            losses.update(loss.item(), inputs.size(0))
            losses_lum_dc_entropy.update(lum_dc_entropy.item(), inputs.size(0))
            losses_chrom_dc_entropy.update(chrom_dc_entropy.item(), inputs.size(0))
            losses_lum_ac_entropy.update(lum_ac_entropy.item(), inputs.size(0))
            losses_chrom_ac_entropy.update(chrom_ac_entropy.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if idx % opt.print_freq == 0:
                print(  'Test: [{0}/{1}]\t'
                        'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                        'Lum_DC {loss_lum_dc_entropy.val:.3f} ({loss_lum_dc_entropy.avg:.3f})\t'
                        'Chrom_DC {loss_chrom_dc_entropy.va1l:.3f} ({loss_chrom_dc_entropy.avg:.3f})\t'
                        'Lum_AC {loss_lum_ac_entropy.val:.3f} ({loss_lum_ac_entropy.avg:.3f})\t'
                        'Chrom_AC {loss_chrom_ac_entropy.val:.3f} ({loss_chrom_ac_entropy.avg:.3f})\t'
                        'Acc@1 {top1.val:.6f} ({top1.avg:.6f})\t'
                        'Acc@5 {top5.val:.6f} ({top5.avg:.6f})'.format(
                        idx, len(val_loader),
                        loss=losses, loss_lum_dc_entropy=losses_lum_dc_entropy, loss_chrom_dc_entropy=losses_chrom_dc_entropy,
                        loss_lum_ac_entropy = losses_lum_ac_entropy, loss_chrom_ac_entropy = losses_chrom_ac_entropy,
                        top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.6f} Acc@5 {top5.avg:.6f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg, outputs, losses_lum_dc_entropy.avg, losses_chrom_dc_entropy.avg, losses_lum_ac_entropy.avg, losses_chrom_ac_entropy.avg


# ====================================================== imagenet ========================================================#

import helper.imagenet_utils as imagenet_utils
import warnings

def train_JPEG(opt, model, criterion, optimizer, train_loader, device, epoch, model_ema=None, scaler=None, backward=True):
    model.train()
    
    metric_logger = imagenet_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", imagenet_utils.SmoothedValue(window_size=1, fmt="{value}"))

    header = f"Train: [{epoch}]"
    for idx, (input, target) in enumerate(metric_logger.log_every(train_loader, opt.print_freq, header)):
        input = input.float()
        input, target = input.to(device), target.to(device)
        
        # ===================forward=====================
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy = model(input)
            loss = criterion(output, target) + opt.weight_lum * (lum_dc_entropy + lum_ac_entropy) + \
                opt.weight_chrom * (chrom_dc_entropy + chrom_ac_entropy)
            acc1, acc5 = imagenet_utils.accuracy(output, target, topk=(1, 5))
            
        # ===================backward=====================
        
        if backward:
            optimizer.zero_grad()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if opt.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if opt.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad_norm)
                optimizer.step()

            if model_ema and idx % opt.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < opt.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

        # ===================meters=====================
        
        batch_size = input.shape[0]
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["losses"].update(loss.item(), n=batch_size)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["lum_dc"].update(lum_dc_entropy.item(), n=batch_size)
        metric_logger.meters["chrom_dc"].update(chrom_dc_entropy.item(), n=batch_size)
        metric_logger.meters["lum_ac"].update(lum_ac_entropy.item(), n=batch_size)
        metric_logger.meters["chrom_ac"].update(chrom_ac_entropy.item(), n=batch_size)

    print(' * Acc@1 {top1.global_avg:.15f} Acc@5 {top5.global_avg:.15f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    
    return metric_logger.acc1.global_avg, metric_logger.losses.global_avg, \
           metric_logger.lum_dc.global_avg, metric_logger.chrom_dc.global_avg, \
           metric_logger.lum_ac.global_avg, metric_logger.chrom_ac.global_avg
    

def evaluate_JPEG(opt, model, criterion, val_loader, device, log_suffix=""):
    # switch to evaluate mode
    model.eval()
    
    metric_logger = imagenet_utils.MetricLogger(delimiter="  ")
    
    header = f"Test: {log_suffix}"
    num_processed_samples = 0
    with torch.inference_mode():
        for idx, data in enumerate(metric_logger.log_every(val_loader, opt.print_freq, header)):
            (input, bpp, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy), target = data
            input = input.float()
            bpp = bpp.float()
            lum_dc_entropy = lum_dc_entropy.float()
            chrom_dc_entropy = chrom_dc_entropy.float()
            lum_ac_entropy = lum_ac_entropy.float()
            chrom_ac_entropy = chrom_ac_entropy.float()
            
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            lum_dc_entropy = lum_dc_entropy.to(device, non_blocking=True).mean() 
            chrom_dc_entropy = chrom_dc_entropy.to(device, non_blocking=True).mean() 
            lum_ac_entropy = lum_ac_entropy.to(device, non_blocking=True).mean() 
            chrom_ac_entropy = chrom_ac_entropy.to(device, non_blocking=True).mean() 
            
            # ===================inference=====================
            
            output = model(input)
            loss = criterion(output, target) + opt.weight_lum * (lum_dc_entropy + lum_ac_entropy) + \
                    opt.weight_chrom * (chrom_dc_entropy + chrom_ac_entropy)
            acc1, acc5 = imagenet_utils.accuracy(output, target, topk=(1, 5))
            
            # ===================meters=====================
            
            batch_size = input.shape[0]
            num_processed_samples += batch_size
            metric_logger.meters["losses"].update(loss.item(), n=batch_size)
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["BPP"].update(bpp.mean().item(), n=batch_size)
            metric_logger.meters["lum_dc"].update(lum_dc_entropy.item(), n=batch_size)
            metric_logger.meters["chrom_dc"].update(chrom_dc_entropy.item(), n=batch_size)
            metric_logger.meters["lum_ac"].update(lum_ac_entropy.item(), n=batch_size)
            metric_logger.meters["chrom_ac"].update(chrom_ac_entropy.item(), n=batch_size)  
        
        print(' * Acc@1 {top1.global_avg:.15f} Acc@5 {top5.global_avg:.15f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))
        
    # gather the stats from all processes
    num_processed_samples = imagenet_utils.reduce_across_processes(num_processed_samples)
    if (hasattr(val_loader.dataset, "__len__") and len(val_loader.dataset) != num_processed_samples and torch.distributed.get_rank() == 0):
        warnings.warn(f"It looks like the dataset has {len(val_loader.dataset)} samples, but {num_processed_samples} samples were used for the validation, which might bias the results. ")

    metric_logger.synchronize_between_processes()

    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, \
           metric_logger.losses.global_avg, metric_logger.BPP.global_avg, \
           metric_logger.lum_dc.global_avg, metric_logger.chrom_dc.global_avg, \
           metric_logger.lum_ac.global_avg, metric_logger.chrom_ac.global_avg


def evaluate_correlation(opt, model, criterion, val_loader, device, log_suffix=""):
    # switch to evaluate mode
    model.eval()
    
    metric_logger = imagenet_utils.MetricLogger(delimiter="  ")
    
    header = f"Test: {log_suffix}"
    num_processed_samples = 0
    with torch.inference_mode():
        for idx, data in enumerate(metric_logger.log_every(val_loader, opt.print_freq, header)):
            (input, bpp, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy), target = data
            input = input.float()
            bpp = bpp.float()
            lum_dc_entropy = lum_dc_entropy.float()
            chrom_dc_entropy = chrom_dc_entropy.float()
            lum_ac_entropy = lum_ac_entropy.float()
            chrom_ac_entropy = chrom_ac_entropy.float()
            
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            lum_dc_entropy = lum_dc_entropy.to(device, non_blocking=True).mean() 
            chrom_dc_entropy = chrom_dc_entropy.to(device, non_blocking=True).mean() 
            lum_ac_entropy = lum_ac_entropy.to(device, non_blocking=True).mean() 
            chrom_ac_entropy = chrom_ac_entropy.to(device, non_blocking=True).mean() 
            
            batch_size = input.shape[0]
            num_processed_samples += batch_size
            if num_processed_samples > 100:
                break
            
            with open("./save/imagenet/mobilenet_v2/BPP_Entropy_correlation_alpha_{}.txt".format(int(opt.JPEG_alpha)), 'a') as log:
                line =  f"{bpp.item():.15f}\t" + \
                        f"{lum_dc_entropy.item():.15f}\t" + \
                        f"{chrom_dc_entropy.item():.15f}\t" + \
                        f"{lum_ac_entropy.item():.15f}\t" + \
                        f"{chrom_ac_entropy.item():.15f}\t" + '\n'
                log.write(line)

    return 0