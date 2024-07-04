from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F

from .util import AverageMeter, accuracy
from .JPEG_layer import *

from torchvision import transforms
# from torchviz import make_dot


def train(epoch, train_loader, model, criterion, optimizer, opt):
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
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy = model(input)
        loss = criterion(output, target) + opt.weight_lum * (lum_dc_entropy + lum_ac_entropy) + \
                opt.weight_chrom * (chrom_dc_entropy + chrom_ac_entropy)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # dot = make_dot(loss, params=dict(model.named_parameters()))
        # dot.format = 'png'
        # dot.render('graph')
        
        losses.update(loss.item(), input.size(0))
        losses_lum_dc_entropy.update(lum_dc_entropy.item(), input.size(0))
        losses_chrom_dc_entropy.update(chrom_dc_entropy.item(), input.size(0))
        losses_lum_ac_entropy.update(lum_ac_entropy.item(), input.size(0))
        losses_chrom_ac_entropy.update(chrom_ac_entropy.item(), input.size(0))
        
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

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
            # (input, bpp, psnr), target = data
            (input, bpp, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy), target = data
            
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                lum_dc_entropy = lum_dc_entropy.cuda().mean() 
                chrom_dc_entropy = chrom_dc_entropy.cuda().mean() 
                lum_ac_entropy = lum_ac_entropy.cuda().mean() 
                chrom_ac_entropy = chrom_ac_entropy.cuda().mean() 

            # compute output
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

            # measure elapsed time
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
                input = input.cuda()
                target = target.cuda()

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
            input = input.cuda()
            target = target.cuda()

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
                inputs = inputs.cuda()
                targets = targets.cuda()

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