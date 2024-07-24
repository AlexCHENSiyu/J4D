import datetime
import os
import time
import json
import helper.presets as presets
import numpy as np
import random
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from helper.JPEG_layer import *
from helper.loops import train_JPEG, evaluate_JPEG
import helper.imagenet_utils as imagenet_utils
from helper.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from helper.transforms import get_mixup_cutmix
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

def create_trial_folder(base_folder):
    trial_num = 1
    while True:
        trial_folder = os.path.join(base_folder, f"trial_{trial_num}")
        if not os.path.exists(trial_folder):
            os.makedirs(trial_folder)
            # print(f"Created folder: {trial_folder}")
            return trial_folder
        trial_num += 1

def parse_option(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    
    # optimization
    parser.add_argument("--batch_size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--num_workers", default=16, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=20, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="initial learning rate")
    parser.add_argument('--lr_decay_epochs', type=str, default='110', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument("--weight_decay", "--wd",  default=1e-4, type=float, metavar="W", help="weight decay", dest="weight_decay")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    
    # model and dataset
    parser.add_argument("--model", default="mobileNet_v2", type=str, choices=['mobileNet_v2','resnet18'], help="model name")
    parser.add_argument("--data_path", default="/home/multicom-pc52/dataset/imagenet", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('--dataset', type=str, default='CUB200', choices=['CUB200','imagenet'], help='dataset')
    parser.add_argument("--output_dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--optimizer", default="adam", choices=['sgd','adam'], type=str, help="optimizer")
    
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
    parser.add_argument('--Q_initial_QF', type=int, default=50, help='Initial Q_table with QF = 50')
    parser.add_argument('--block_size', type=int, default=8, help='the experiment id')
    parser.add_argument('--num_bit', type=int, default=11, help='Number of bits to represent DCT coeff')
    parser.add_argument('--min_Q_Step', type=float, default=1.0, help='Minumum Quantization Step')
    parser.add_argument('--max_Q_Step', type=float, default=255, help='Maximum Quantization Step')
    parser.add_argument('--num_non_zero_q', type=int, default=5, choices=range(2,2**10 - 1), help='Window size for the reconstruction space')
    parser.add_argument('--weight_lum', type=float, default=0.0, help='weight for lum entropy')
    parser.add_argument('--weight_chrom', type=float, default=0.0, help='weight for Chrom entropy')
    
    parser.add_argument('--seed', type=int, default=42, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of micro-batches to accumulate')
    
    # other parameter for datasets
    parser.add_argument("--norm_weight_decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias_weight_decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer_embedding_decay", default=None, type=float,help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup_alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix_alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr_scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr_warmup_epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr_warmup_method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr_step_size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr_gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr_min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache_dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync_bn", dest="sync_bn", help="Use sync batch norm", action="store_true",)
    parser.add_argument("--test_only", dest="test_only", help="Only test the model", action="store_true",)
    parser.add_argument("--auto_augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra_magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix_severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random_erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model_ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model_ema_steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model_ema_decay", type=float, default=0.99998,help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use_deterministic_algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val_resize_size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val_crop_size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train_crop_size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip_grad_norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra_sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra_reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use_v2", action="store_true", help="Use V2 transforms")
    
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
            opt.model_name = 'JPEG_lr_{}_hardness_{}_weight_{}_{}'.format(opt.JPEG_learning_rate, opt.hardness, opt.weight_lum, opt.weight_chrom)

        if opt.QF_Inital:
            opt.model_name += '_QF_{}'.format(opt.Q_initial_QF)
    else:
        opt.added_layer = "vanilla"
        opt.model_name = '{}'.format(opt.model)
    
    opt.output_dir = './save/{}/{}'.format(opt.dataset, opt.model)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    base_folder = os.path.join(opt.output_dir, opt.model_name)
    opt.trial_folder = create_trial_folder(base_folder)

    # 1     2        3        4         5         6          7   8           9             
    # epoch best_acc test_acc train_acc test_loss train_loss bpp test_lum_dc test_chrom_dc 
    # 10          11            12           13             14           15             16    17   
    # test_lum_ac test_chrom_ac train_lum_dc train_chrom_dc train_lum_ac train_chrom_ac q_min q_max
    opt.logs_fname = os.path.join(opt.trial_folder, "logs.txt")
    with open(opt.logs_fname, 'a') as log:
        line =  'ep' + "\t" + \
                'best_acc' + "\t\t\t" + 'test_acc' + "\t\t\t" + 'train_acc' + "\t\t\t" +  \
                'test_loss' + "\t\t\t" + 'train_loss' + "\t\t\t" + \
                'bpp' + "\t\t\t\t\t" + \
                'test_lum_dc' + "\t\t\t" + 'test_chrom_dc' + "\t\t" + 'test_lum_ac' + "\t\t\t" + 'test_chrom_ac' + "\t\t" + \
                'train_lum_dc' + "\t\t" + 'train_chrom_dc' + "\t\t" + 'train_lum_ac' + "\t\t" + 'train_chrom_ac' + "\t\t" + \
                'q_min' + "\t\t\t\t" + 'q_max' + '\t\t' + 'alpha_min' + "\t\t\t" + 'alpha_max' + '\n'
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


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, opt):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (opt.val_resize_size, opt.val_crop_size, opt.train_crop_size,)
    interpolation = InterpolationMode(opt.interpolation)

    print("Loading training data")
    start_time = time.time()
    cache_path = _get_cache_path(traindir)
    if opt.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(opt, "auto_augment", None)
        random_erase_prob = getattr(opt, "random_erase", 0.0)
        ra_magnitude = getattr(opt, "ra_magnitude", None)
        augmix_severity = getattr(opt, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=opt.backend,
                use_v2=opt.use_v2,),)
        
        if opt.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            imagenet_utils.mkdir(os.path.dirname(cache_path))
            imagenet_utils.save_on_master((dataset, traindir), cache_path)
    
    print("Took", time.time() - start_time)


    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if opt.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if opt.weights and opt.test_only:
            weights = torchvision.models.get_weight(opt.weights)
            preprocessing = weights.transforms(antialias=True)
            if opt.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=opt.backend,
                use_v2=opt.use_v2,)

        dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing,)
        if opt.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            imagenet_utils.mkdir(os.path.dirname(cache_path))
            imagenet_utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if opt.distributed:
        if hasattr(opt, "ra_sampler") and opt.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=opt.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(opt):
    best_acc = 0
    best_bpp = 0

    if opt.world_size == 1:
        # Non-distributed mode
        print("Running in non-distributed mode.")
        opt.distributed = False
    else:
        # Distributed mode
        imagenet_utils.init_distributed_mode(opt)
    print(opt)
    
    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if opt.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device(opt.device)
    
    # load dataset
    train_dir = os.path.join(opt.data_path, "train")
    val_dir = os.path.join(opt.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, opt)

    # data arguementation
    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(mixup_alpha=opt.mixup_alpha, cutmix_alpha=opt.cutmix_alpha, num_classes=num_classes, use_v2=opt.use_v2)
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    # data_loader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler, num_workers=opt.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, sampler=test_sampler, num_workers=opt.num_workers, pin_memory=True)

    # load underlying model
    print("loading model {}".format(opt.model))
    if opt.model == "mobileNet_v2":
        underlying_model = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1, progress=True)
    elif opt.model == "resnet18":
        underlying_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
    else:
        raise RuntimeError(f"Invalid model {opt.model}. Only mobileNet_v2, resnet18 are supported.")
    # underlying_model = torchvision.models.get_model(opt.model, weights=opt.weights, num_classes=num_classes)
    
    # load whole model
    opt_name = opt.optimizer.lower()
    if opt.JPEG_enable:
        jpeg_layer = JPEG_layer(opt=opt, img_shape=(224, 224, 3), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))        
        model = CustomModel(jpeg_layer, underlying_model)
        
        # based on optimizer
        if opt_name.startswith("sgd"):
            optimizer_data = [{'params': model.jpeg_layer.lum_qtable, 'lr': opt.JPEG_learning_rate, 'momentum': opt.momentum,},
                              {'params': model.jpeg_layer.chrom_qtable, 'lr': (opt.JPEG_learning_rate), 'momentum': opt.momentum,}]
        elif opt_name == "adam":
            optimizer_data = [{'params': model.jpeg_layer.lum_qtable, }, {'params': model.jpeg_layer.chrom_qtable, }]
        else:
            raise RuntimeError(f"Invalid optimizer {opt.optimizer}. Only SGD, AdamW are supported.")
      
        if opt.JPEG_alpha_trainable:
            optimizer_data.append({'params': model.jpeg_layer.alpha_lum, 'lr': opt.alpha_learning_rate, 'momentum': opt.momentum,})
            optimizer_data.append({'params': model.jpeg_layer.alpha_chrom, 'lr': (opt.alpha_learning_rate)/np.sqrt(2), 'momentum': opt.momentum,})
    else:
        model = underlying_model
        optimizer_data = [{'params': model.parameters(), 'lr': opt.learning_rate, 'momentum': opt.momentum, 'weight_decay': opt.weight_decay},]
    
    if opt.distributed and opt.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)

    # optimizer
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(optimizer_data)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(optimizer_data, lr=1e-2, betas=(0.9, 0.999), eps=1e-08)
    else:
        raise RuntimeError(f"Invalid optimizer {opt.optimizer}. Only SGD, RMSprop and AdamW are supported.")

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)

    # weight decay
    # custom_keys_weight_decay = []
    # if args.bias_weight_decay is not None:
    #     custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    # if args.transformer_embedding_decay is not None:
    #     for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
    #         custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    # parameters = utils.set_weight_decay(
    #     model, args.weight_decay, norm_weight_decay=args.norm_weight_decay,
    #     custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,)

    # GradScaler
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    # lr scheduler
    opt.lr_scheduler = opt.lr_scheduler.lower()
    if opt.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
    elif opt.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs - opt.lr_warmup_epochs, eta_min=opt.lr_min)
    elif opt.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_gamma)
    else:
        raise RuntimeError(f"Invalid lr scheduler '{opt.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR " "are supported.")

    # set learning rate warmup
    if opt.lr_warmup_epochs > 0:
        if opt.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=opt.lr_warmup_decay, total_iters=opt.lr_warmup_epochs)
        elif opt.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=opt.lr_warmup_decay, total_iters=opt.lr_warmup_epochs)
        else:
            raise RuntimeError(f"Invalid warmup lr method '{opt.lr_warmup_method}'. Only linear and constant are supported.")
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[opt.lr_warmup_epochs])
    else:
        lr_scheduler = main_lr_scheduler

    # data parallel
    model_without_ddp = model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
        model_without_ddp = model.module

    # model ema
    model_ema = None
    if opt.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = opt.world_size * opt.batch_size * opt.model_ema_steps / opt.epochs
        alpha = 1.0 - opt.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = imagenet_utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    
    # resume training
    if opt.resume:
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not opt.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        opt.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    # function for validation inference
    # HDQ_preprocess = resize_compression(HDQ_transforms(QF_Y=10, QF_C=10))
    HDQ_preprocess = resize_compression(JPEG_layer_transforms(opt, img_shape=(224, 224, 3), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    val_loader.dataset.transform = HDQ_preprocess
    
    # routine
    if opt.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate_JPEG(model_ema, criterion, val_loader, device=device, log_suffix="EMA")
        else:
            evaluate_JPEG(model, criterion, val_loader, device=device)
        return

    # start traning
    print("Start training")
    for epoch in range(opt.start_epoch, opt.epochs+1):
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        
        # train
        print("\n==> training...")
        time1 = time.time()
        if epoch == 0:
            # train with default settings (Validation)
            train_acc, train_loss, train_lum_dc, train_chrom_dc, train_lum_ac, train_chrom_ac = train_JPEG(opt, model, criterion, optimizer, train_loader, device, epoch, opt, model_ema, scaler, backward=False)
        else:
            # train
            train_acc, train_loss, train_lum_dc, train_chrom_dc, train_lum_ac, train_chrom_ac = train_JPEG(opt, model, criterion, optimizer, train_loader, device, epoch, opt, model_ema, scaler)
        lr_scheduler.step()
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
        test_acc, test_acc_top5, test_loss, bpp, test_lum_dc, test_chrom_dc, test_lum_ac, test_chrom_ac = evaluate_JPEG(opt, model, criterion, val_loader, device=device)
        time2 = time.time()
        print('Epoch [{}]: total time {:.2f} s'.format(epoch, time2 - time1))
        
        # record best acc
        if test_acc > best_acc:
            best_acc = test_acc
            best_bpp = bpp
        print('Best Accuracy: {:.2f} % \t BPP: {:.5f} '.format(best_acc, best_bpp))
        
        if model_ema:
            test_acc, test_acc_top5, test_loss, bpp, test_lum_dc, test_chrom_dc, test_lum_ac, test_chrom_ac = evaluate_JPEG(model_ema, criterion, val_loader, device=device, log_suffix="EMA")
        
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
    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": opt,}
    
    if model_ema:
        checkpoint["model_ema"] = model_ema.state_dict()
    if scaler:
        checkpoint["scaler"] = scaler.state_dict()
    imagenet_utils.save_on_master(checkpoint, os.path.join(opt.trial_folder, "model_last.pth"))
    
    # After training, save the final q_table and alpha
    print("\n==> Finish training: ", opt.model_name)
    print('Best accuracy:', best_acc.item())


if __name__ == "__main__":
    timeStart = time.time()
    
    opt = parse_option()
    main(opt)
    
    timeEnd = time.time()
    print('\nTotal time {:.2f} s'.format(timeEnd - timeStart))