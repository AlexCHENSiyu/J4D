from __future__ import print_function, division
import torch
from .JPEG_layer import *
import helper.imagenet_utils as imagenet_utils
import warnings


def train_JPEG(opt, model, criterion, optimizer, train_loader, device, epoch, scaler=None, backward=True):
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
            
            with open("./save/imagenet/mobilenet_v2/ours_BPP_Entropy_correlation_alpha_{}.txt".format(int(opt.JPEG_alpha)), 'a') as log:
                line =  f"{bpp.mean().item():.15f}\t" + \
                        f"{lum_dc_entropy.item():.15f}\t" + \
                        f"{chrom_dc_entropy.item():.15f}\t" + \
                        f"{lum_ac_entropy.item():.15f}\t" + \
                        f"{chrom_ac_entropy.item():.15f}\t" + '\n'
                log.write(line)

    return 0