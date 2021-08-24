import IPython
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms


MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

def get_loaders(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    train_dataset = CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    indices = list(range(len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[:-5000]),
        pin_memory=True,
        num_workers=0,
        shuffle=True
    )

    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[-5000:]),
        pin_memory=True,
        num_workers=0,
    )

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    valid_dataset = CIFAR10(
        root=args.data,
        train=False,
        download=False,
        transform=valid_transform,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    #repeat_train_loader = RepeatedDataLoader(train_loader)
    repeat_reward_loader = RepeatedDataLoader(reward_loader)
    repeat_valid_loader = RepeatedDataLoader(valid_loader)

    return train_loader, repeat_reward_loader, repeat_valid_loader, valid_loader


class RepeatedDataLoader():

    def __init__(self, data_loader, debug=False):
        self.data_loader = data_loader
        self.data_iter = None
        # self.debug = debug
        # print("reset dataloader. ", debug)
        self.failure_counter = 0

    def __len__(self):
        return len(self.data_loader)

    def next_batch(self):
        if self.data_iter is None:
            self.data_iter = iter(self.data_loader)
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        except RuntimeError as e:
            print(e)
            # re-initialize the data iterator,
            self.failure_counter += 1
            # IPython.embed()
            self.data_iter = iter(self.data_loader)
            if self.failure_counter > 5:
                raise e
            batch = self.next_batch()
            self.failure_counter = 0

        return batch

#
# class RepeatedDataLoader(DataLoader):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Initialize an iterator over the dataset.
#         self.dataset_iterator = super().__iter__()
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         try:
#             batch = next(self.dataset_iterator)
#         except StopIteration:
#             # Dataset exhausted, use a new fresh iterator.
#             self.dataset_iterator = super().__iter__()
#             batch = next(self.dataset_iterator)
#         return batch
#
#     def next_batch(self):
#         return self.__next__()
