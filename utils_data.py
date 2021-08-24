import os
import copy
import logging
import shutil
import threading
import gc
import zipfile
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def convert_to_pil(bytes_obj):
    img = Image.open(BytesIO(bytes_obj))
    return img.convert('RGB')


class ReadImageThread(threading.Thread):
    def __init__(self, root, fnames, class_id, target_list):
        threading.Thread.__init__(self)
        self.root = root
        self.fnames = fnames
        self.class_id = class_id
        self.target_list = target_list
        
    def run(self):
        for fname in self.fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(self.root, fname)
                with open(path, 'rb') as f:
                    image = f.read()
                item = (image, self.class_id)
                self.target_list.append(item)


class InMemoryDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.samples = []
        classes, class_to_idx = self.find_classes(self.path)
        dir = os.path.expanduser(self.path)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if num_workers == 1:
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            with open(path, 'rb') as f:
                                image = f.read()
                            item = (image, class_to_idx[target])
                            self.samples.append(item)
                else:
                    fnames = sorted(fnames)
                    num_files = len(fnames)
                    threads = []
                    res = [[] for i in range(num_workers)]
                    num_per_worker = num_files // num_workers
                    for i in range(num_workers):
                        start_index = num_per_worker * i
                        end_index = num_files if i == num_workers - 1 else num_per_worker * (i+1)
                        thread = ReadImageThread(root, fnames[start_index:end_index], class_to_idx[target], res[i])
                        threads.append(thread)
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    for item in res:
                        self.samples += item
                    del res, threads
                    gc.collect()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def find_classes(root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    

class ZipDataset(data.Dataset):
    def __init__(self, path, transform=None, prefix=''):
        super(ZipDataset, self).__init__()
        self.prefix = prefix
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        with zipfile.ZipFile(self.path, 'r') as reader:
            classes, class_to_idx = self.find_classes(reader)
            fnames = sorted(reader.namelist())
        for fname in fnames:
            if self.is_directory(fname) or self.ignore(fname):
                continue
            target = self.get_target(fname)
            item = (fname, class_to_idx[target])
            self.samples.append(item)
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        with zipfile.ZipFile(self.path, 'r') as reader:
            sample = reader.read(sample)
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def ignore(self, fname):
        if 'preprocessed.pkl' in fname:
            return True
        return False

    def is_directory(self, fname):
        if fname.startswith(self.prefix) and fname.endswith('/'):
            return True
        return False
    
    def get_target(self, fname):
        assert fname.startswith(f'{self.prefix}'), print(f'get_target error: {fname} does not start with train/n')
        return fname.split('/')[1]
    
    def find_classes(self, reader):
        classes = [self.get_target(name) for name in reader.namelist() if self.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ReadZipImageThread(threading.Thread):
    def __init__(self, reader, fnames, class_to_idx, target_list):
        threading.Thread.__init__(self)
        self.reader = reader
        self.fnames = fnames
        self.target_list = target_list
        self.class_to_idx = class_to_idx
    
    def run(self):
        for fname in self.fnames:
            if InMemoryZipDataset.is_directory(fname):
                continue
            image = self.reader.read(fname)
            class_id = self.class_to_idx[InMemoryZipDataset.get_target(fname)]
            item = (image, class_id)
            self.target_list.append(item)


class InMemoryZipDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        reader = zipfile.ZipFile(self.path, 'r')
        classes, class_to_idx = self.find_classes(reader)
        fnames = sorted(reader.namelist())
        if num_workers == 1:
            for fname in fnames:
                if self.is_directory(fname):
                    continue
                target = self.get_target(fname)
                image = reader.read(fname)
                item = (image, class_to_idx[target])
                self.samples.append(item)
        else:
            num_files = len(fnames)
            threads = []
            res = [[] for i in range(num_workers)]
            num_per_worker = num_files // num_workers
            for i in range(num_workers):
                start_index = num_per_worker * i
                end_index = num_files if i == num_workers - 1 else (i+1) * num_per_worker
                thread = ReadZipImageThread(reader, fnames[start_index:end_index], class_to_idx, res[i])
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for item in res:
                self.samples += item
            del res, threads
            gc.collect()
        reader.close()
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]

    @staticmethod
    def find_classes(fname):
        classes = [ZipDataset.get_target(name) for name in fname.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


# class WrappedDataset(data.Dataset):
#     def __init__(self, dataset, indices, transform=None):
#         super(WrappedDataset, self).__init__()
#         self.dataset = dataset
#         self.indices = indices
#         self.transform = transform
#         self.target_transform = None
#         self.samples = []
            
#     def __len__(self):
#         return len(self.indices)
    
#     def __getitem__(self, index):
#         assert index in self.indices
#         sample, target = self.dataset[index]
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample, target
    
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Dataset: {}\n'.format(self.dataset)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


# class ControllerDataset(torch.utils.data.Dataset):
#     def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
#         super(ControllerDataset, self).__init__()
#         if targets is not None:
#             assert len(inputs) == len(targets)
#         self.inputs = copy.deepcopy(inputs)
#         self.targets = copy.deepcopy(targets)
#         self.train = train
#         self.sos_id = sos_id
#         self.eos_id = eos_id
    
#     def __getitem__(self, index):
#         encoder_input = self.inputs[index]
#         encoder_target = None
#         if self.targets is not None:
#             encoder_target = [self.targets[index]]
#         if self.train:
#             decoder_input = [self.sos_id] + encoder_input[:-1]
#             sample = {
#                 'encoder_input': torch.LongTensor(encoder_input),
#                 'encoder_target': torch.FloatTensor(encoder_target),
#                 'decoder_input': torch.LongTensor(decoder_input),
#                 'decoder_target': torch.LongTensor(encoder_input),
#             }
#         else:
#             sample = {
#                 'encoder_input': torch.LongTensor(encoder_input),
#                 'decoder_target': torch.LongTensor(encoder_input),
#             }
#             if encoder_target is not None:
#                 sample['encoder_target'] = torch.FloatTensor(encoder_target)
#         return sample
    
#     def __len__(self):
#         return len(self.inputs)

