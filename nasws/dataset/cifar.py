import numpy as np
import utils
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_supernet_cifar10(args, shuffle_test=False, debug_imgnet=False):
    if debug_imgnet:
        train_transform, valid_transform = utils.debug_data_transforms_cifar10_like_imagenet(
            args.cutout_length if args.cutout else None)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(
            args.cutout_length if args.cutout else None)
    timeout = 0
    train_data = CIFAR10(root=args.dataset_dir, train=True, download=True, transform=train_transform)
    test_data = CIFAR10(root=args.dataset_dir, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))

    if args.train_portion < 1.:
        train_queue = DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=args.n_worker, timeout=timeout)

        valid_queue = DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=args.n_worker,timeout=timeout)
    else:
        train_queue = DataLoader(
            train_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=args.n_worker, timeout=timeout)

        valid_queue = DataLoader(
            test_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=args.n_worker,timeout=timeout)
        
    test_queue = DataLoader(
        test_data, batch_size=args.evaluate_batch_size,
        shuffle=shuffle_test, pin_memory=True, num_workers=args.n_worker,timeout=timeout)
    return train_queue, valid_queue, test_queue
