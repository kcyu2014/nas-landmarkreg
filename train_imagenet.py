import os
import sys
import numpy as np
import time
import torch
import glob
import random
import logging
import argparse
import tensorflow as tf
tf.enable_eager_execution()

import utils
import utils_nvidia
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from nasws.cnn.policy.darts_policy.genotypes import *
from nasws.cnn.utils import AverageMeter


parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
# parser.add_argument('--early_stop_epoch', type=int, default=-1, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--save_every_epoch', type=int, default=20, help='save periodically')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--base_model', type=str, default='DARTS', help='define the base model structure')
parser.add_argument('--arch_method', type=str, default='PCDARTS', help='method to generate the arch.')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='run', help='note for this run')
parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
parser.add_argument('--resume', default=False, action='store_true', help='resume training')
parser.add_argument('--resume_epoch', type=int, default=None, help='resume epochs')
parser.add_argument('--zip_dataset', default=False, action='store_true', help='Using zip datasets to see speed up.')
parser.add_argument('--inmemory_dataset', default=False, action='store_true', help='load datasets to memroy to see speed up.')
parser.add_argument('--webdataset', default=False, action='store_true', help='Use WebDataset to load imagenet...')

parser.add_argument('--apex_enable', default=False, action='store_true', help='enabling apex to spped up')
parser.add_argument('--apex_profiling', default=-1, type=int, help='Profile the apex to see if it improves the result.')
parser.add_argument('--apex_local_rank', default=0, type=int, help='Local rank')
parser.add_argument('--dali_enable', default=False, action='store_true', help='enabling dali to spped up')
parser.add_argument('--dali_cpu', default=False, action='store_true', help='dali cpu based version')
parser.add_argument('--dali_profiling', default=-1, type=int, help='Profile the apex to see if it improves the result.')

args, unparsed = parser.parse_known_args()

args.save = '{}-{}'.format(args.save, args.note)

try:
    utils.create_exp_dir(args.save)
    utils.save_json(args, args.save + '/args.json')
except FileExistsError as e:
    print("File exists, try to reload later.")


logger = utils.get_logger(
    'ImageNet Reproduce.',
    file_handler=utils.get_file_handler(os.path.join(args.save, 'log.txt'))
)
writer = SummaryWriter(args.save + '/runs')
CLASSES = 1000

def _tensorboard(epoch, loss, lr, acc, prefix='Train'):
    # Tensorboard
    writer.add_scalar(f'{prefix}/loss',
                            loss,
                            epoch)
    writer.add_scalar(f'{prefix}/accuracy',
                            acc,
                            epoch)
    if prefix.lower() == 'train':
        writer.add_scalar('lr ', lr, epoch)


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss 


def parse_darts_arch(args):
    return utils.arch_to_genotype(args.arch, args.arch_method)
    # if args.arch.lower() == 'pc-darts-image':
    #     return PC_DARTS_image
    # elif args.arch.lower() == 'pc-darts-cifar':
    #     return PC_DARTS_cifar
    # elif args.arch.lower() == 'darts-v2':
    #     return DARTS_V2
    # else:
    #     return eval(args.arch)


def get_model(args):
    if args.base_model == 'DARTS':
        genotype = parse_darts_arch(args)
        logging.info('---------Genotype---------')
        logging.info(genotype)
        logging.info('--------------------------') 
        from nasws.cnn.search_space.darts.model import NetworkImageNet
        model = NetworkImageNet(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    else:
        raise NotImplementedError(f'Base model not yet supported {args.base_model}')
    return model


def load_imagenet_webdataset():
    import webdataset as wds
    crop_size = 224
    trainsize = 1281167

    data_dir = args.data_dir
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.apex_enable:
        import utils_nvidia
        collate_fn = lambda b: utils_nvidia.fast_collate(b, torch.contiguous_format)
    else:
        collate_fn = None

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
            ])
    valid_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])
    def identity(x):
        return x

    def worker_urls(urls):
        result = wds.worker_urls(urls)
        print("worker_urls returning", len(result), "of", len(urls), "urls", file=sys.stderr)
        return result

    def make_train_loader_wds(args):
        print("=> using WebDataset loader")
        num_batches = trainsize // args.batch_size
        webdataset_folder = '/data/kyu/ILSVRC2012-shards' if os.path.exists('/data/kyu/ILSVRC2012-shards') else '/cvlabsrc1/cvlab/datasets_kyu/ILSVRC2012-shards'
        train_dataset = (
            wds.Dataset(webdataset_folder + '/imagenet-train-{000000..00281}.tar', length=num_batches, shard_selection=worker_urls)
            # wds.Dataset('/cvlabsrc1/cvlab/datasets_kyu/ILSVRC2012-shards/imagenet-train-{000000..001281}.tar', length=num_batches, shard_selection=worker_urls)
            .shuffle(True)
            .decode("pil")
            .to_tuple("jpg;png;jpeg cls")
            .map_tuple(train_transform, identity)
            .batched(args.batch_size)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=None, shuffle=False, num_workers=args.workers,
        )
        return train_loader

    def make_val_loader(args):
        val_root = '/data/kyu/ILSVRC2012' if os.path.exists('/data/kyu/ILSVRC2012') else '/cvlabsrc1/cvlab/datasets_kyu/ILSVRC2012'
        valdir = os.path.join(val_root, "val")
        val_dataset = dset.ImageFolder(valdir, valid_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        return val_loader

    return make_train_loader_wds(args), make_val_loader(args)

def load_imagenet():
    crop_size = 224

    data_dir = args.data_dir
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.apex_enable:
        collate_fn = lambda b: utils_nvidia.fast_collate(b, torch.contiguous_format)
    else:
        collate_fn = None

    if args.dali_enable and not args.zip_dataset:
        pipe = utils_nvidia.HybridTrainPipe(batch_size=args.batch_size,
                           num_threads=args.workers,
                           device_id=args.apex_local_rank,
                           data_dir=traindir,
                           crop=crop_size,
                           dali_cpu=args.dali_cpu,
                           shard_id=args.apex_local_rank,
                           num_shards=args.world_size,
                           args=args)
        pipe.build()
        train_queue = utils_nvidia.DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

        pipe = utils_nvidia.HybridValPipe(batch_size=args.batch_size,
                            num_threads=args.workers,
                            device_id=args.apex_local_rank,
                            data_dir=validdir,
                            crop=crop_size,
                            size=256,
                            shard_id=args.apex_local_rank,
                            num_shards=args.world_size, 
                            args=args)
        pipe.build()
        valid_queue = utils_nvidia.DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))  

    else:
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
        valid_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    normalize,
                ])
        
        import utils_data
        if args.zip_dataset:
        
            traindir = os.path.join(args.data_dir, 'train.zip')
            valdir = os.path.join(args.data_dir, 'valid.zip')
            if args.inmemory_dataset:
                collate_fn = None
                train_data = utils_data.InMemoryZipDataset(traindir, transform=train_transform, num_workers=args.workers)
                valid_data = utils_data.InMemoryZipDataset(valdir, transform=valid_transform, num_workers=args.workers)

            else:
                train_data = utils_data.ZipDataset(traindir, transform=train_transform, prefix='train')
                valid_data = utils_data.ZipDataset(valdir, transform=valid_transform, prefix='val')
            
        else:
            if args.inmemory_dataset:
                logging.info('Using in-memory dataset:')
                train_data = utils_data.InMemoryDataset(traindir, num_workers=args.workers)
                valid_data = utils_data.InMemoryDataset(validdir, num_workers=args.workers)
                logging.info('Finish loading')
            else:
                train_data = dset.ImageFolder(
                    traindir,
                    train_transform)

                valid_data = dset.ImageFolder(
                    validdir,
                    valid_transform)
    
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, 
            collate_fn=collate_fn)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers,
            collate_fn=collate_fn)

    return train_queue, valid_queue


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    model = get_model(args)

    if args.apex_enable:
        args.distributed = False
        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.gpu = num_gpus
        args.world_size = 1
        args.learning_rate = args.learning_rate*float(args.batch_size*args.world_size)/1024
        model = model.cuda().to(memory_format=torch.contiguous_format)

    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    if args.apex_enable: 
        import apex.amp as amp
        model, optimizer = amp.initialize(
            model, optimizer,
            opt_level='O1', # official mixed precision
            # keep_batchnorm_fp32=True, # bn 32 to accelarate further.
            loss_scale=None)    # do not scale
        
        args.apex_opt_level='O1'

    if num_gpus > 1:
        model = nn.DataParallel(model)

    if args.webdataset:
        train_queue, valid_queue = load_imagenet_webdataset()
    else:
        train_queue, valid_queue = load_imagenet()

#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0
    init_epoch = 0
    if args.resume:
        try:
            state = utils.load_checkpoint_v2(args.save, epoch=args.resume_epoch)
            init_epoch = state['misc']['epoch'] + 1
            best_acc = state['misc']['best_acc_top1']
            try:
                model.load_state_dict(state['model'])
            except RuntimeError as e:
                model = nn.DataParallel(model)
                model.load_state_dict(state['model'])
                model = model.module
            
            optimizer.load_state_dict(state['optimizer'])
            if 'scheduler_state' in state.keys():
                logging.info('Resuming LR Scheduler state-dict ...')
                scheduler.load_state_dict(state['scheduler_state'])
            if args.apex_enable:
                logging.info('Resuming amp state-dict ...')
                amp.load_state_dict(state['amp'])
            
            logging.info(
                f"Resuming from previous checkpoint. At epoch {init_epoch}, best accuracy {best_acc}")
        except FileNotFoundError as e:
            logging.info('did not find any check point. training from scratch...')
    
    for epoch in range(init_epoch, args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            logging.info('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
            current_lr = current_lr * (epoch + 1) / 5.0
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        if args.apex_enable:
            if args.dali_enable:
                train_acc, train_obj = utils_nvidia.dali_apex_train(
                    train_queue, model, criterion_smooth, optimizer, epoch, args)
            else:
                train_acc, train_obj = utils_nvidia.train(train_queue, model, criterion_smooth, optimizer, epoch, args)
        else:
            train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)

        logging.info('Train_acc: %f', train_acc)
        if args.apex_enable:
            if args.dali_enable:
                valid_acc_top1, valid_acc_top5, valid_obj = utils_nvidia.dali_validate(valid_queue, model, criterion, args)
            else:
                valid_acc_top1, valid_acc_top5, valid_obj = utils_nvidia.validate(valid_queue, model, criterion, args)
        else:
            valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)

        _tensorboard(epoch, train_obj, current_lr, train_acc, prefix='Train')
        _tensorboard(epoch, valid_obj, current_lr, valid_acc_top1, prefix='Valid_top1')
        _tensorboard(epoch, valid_obj, current_lr, valid_acc_top5, prefix='Valid_top5')

        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        
        utils.save_checkpoint(model, optimizer, 
            running_stats={
            'epoch': epoch + 1,
            'best_acc_top1': best_acc_top1,
            }, 
            path=args.save,
            scheduler=scheduler,
            amp=amp if args.apex_enable else None, 
            backup_weights=epoch % args.save_every_epoch == 0
            )
        if args.dali_enable:
            train_queue.reset()
            valid_queue.reset()
        
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    threshold = args.epochs - 245
    if args.epochs -  epoch > threshold:
        lr = args.learning_rate * (args.epochs - threshold - epoch) / (args.epochs - threshold)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - threshold) * threshold)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def train(train_queue, model, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        if args.debug and step > 100:
            logging.warning('Break after 100 batch')
            break
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs', 
                                    step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        if args.debug and step > 100:
            logging.warning('Break after 100 batch')
            break

        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main() 