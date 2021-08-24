import os
import utils
import torch
import logging
import IPython
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

ZIP_DATASET = False
INMEMORY_DATASET = False


def create_imagenet_file_split(root_dir, portion, indices_only=False, valid_portion=-1, dataset=None):
    """ For dali pipeline usage """
    dataset = dataset or dset.ImageFolder(root_dir)
    num_dataset = len(dataset)

    indices = list(range(num_dataset))
    np.random.shuffle(indices)    
    split = int(np.floor(portion * num_dataset))
    train_idx, val_idx = indices[:split], indices[split:] 
    if valid_portion > 0:
        v_split = int(np.floor(valid_portion * num_dataset))
        val_idx = indices[:v_split]

    if indices_only:
        return train_idx, val_idx
    
    train_files = [(dataset.imgs[i][0].split(root_dir)[1], dataset.imgs[i][1]) for i in train_idx]
    valid_files = [(dataset.imgs[i][0].split(root_dir)[1], dataset.imgs[i][1]) for i in val_idx]
    return train_files, valid_files


def load_supernet_imagenet(args):
    crop_size = 224

    data_dir = args.dataset_dir
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.apex_enable:
        import utils_nvidia
        collate_fn = lambda b: utils_nvidia.fast_collate(b, torch.contiguous_format)
    else:
        collate_fn = None

    if args.dali_enable and not ZIP_DATASET:
        import utils_nvidia
        if args.train_portion == 1:
            pipe = utils_nvidia.HybridTrainPipe(batch_size=args.batch_size,
                            num_threads=args.n_worker,
                            device_id=args.apex_local_rank,
                            data_dir=traindir,
                            crop=crop_size,
                            dali_cpu=args.dali_cpu,
                            shard_id=args.apex_local_rank,
                            num_shards=args.world_size,
                            args=args)
            pipe.build()
            train_queue = utils_nvidia.DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
            train_valid_queue = None
        else:
            train_files, val_files = create_imagenet_file_split(traindir, args.train_portion, False, args.valid_portion)

            pipe = utils_nvidia.HybridTrainPipe(batch_size=args.batch_size,
                            num_threads=args.n_worker,
                            device_id=args.apex_local_rank,
                            data_dir=traindir,
                            crop=crop_size,
                            dali_cpu=args.dali_cpu,
                            shard_id=args.apex_local_rank,
                            num_shards=args.world_size,
                            args=args,
                            file_list=train_files)
            pipe.build()
            train_queue = utils_nvidia.DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

            pipe = utils_nvidia.HybridTrainPipe(batch_size=args.batch_size,
                            num_threads=args.n_worker,
                            device_id=args.apex_local_rank,
                            data_dir=traindir,
                            crop=crop_size,
                            dali_cpu=args.dali_cpu,
                            shard_id=args.apex_local_rank,
                            num_shards=args.world_size,
                            args=args,
                            file_list=val_files)
            pipe.build()
            train_valid_queue = utils_nvidia.DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

        pipe = utils_nvidia.HybridValPipe(batch_size=args.batch_size,
                            num_threads=args.n_worker,
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
        if ZIP_DATASET:
        
            traindir = os.path.join(args.data_dir, 'train.zip')
            valdir = os.path.join(args.data_dir, 'valid.zip')
            if INMEMORY_DATASET:
                collate_fn = None
                train_data = utils_data.InMemoryZipDataset(traindir, transform=train_transform, num_workers=args.n_worker)
                valid_data = utils_data.InMemoryZipDataset(valdir, transform=valid_transform, num_workers=args.n_worker)

            else:
                train_data = utils_data.ZipDataset(traindir, transform=train_transform, prefix='train')
                valid_data = utils_data.ZipDataset(valdir, transform=valid_transform, prefix='val')
            
        else:
            if INMEMORY_DATASET:
                logging.info('Using in-memory dataset:')
                train_data = utils_data.InMemoryDataset(traindir, num_workers=args.n_worker)
                valid_data = utils_data.InMemoryDataset(validdir, num_workers=args.n_worker)
                logging.info('Finish loading')
            else:
                train_data = dset.ImageFolder(
                    traindir,
                    train_transform)

                valid_data = dset.ImageFolder(
                    validdir,
                    valid_transform)

        if args.train_portion < 1 or args.valid_portion < 1:
            # load the index and create tr sampler...
            tr_idx, val_idx = create_imagenet_file_split(
                traindir, args.train_portion, True, args.valid_portion, dataset=train_data)
            tr_sampler = torch.utils.data.sampler.SubsetRandomSampler(tr_idx)
            tr_val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
        else:
            tr_sampler, tr_val_sampler = None, None
        
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=not tr_sampler, pin_memory=True, num_workers=args.n_worker, 
            collate_fn=collate_fn, 
            sampler=tr_sampler)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.n_worker,
            collate_fn=collate_fn)
        if tr_val_sampler:
            train_valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.n_worker, 
            collate_fn=collate_fn,
            sampler=tr_val_sampler
        )
        else:
            train_valid_queue = None
    
    return train_queue, train_valid_queue, valid_queue


