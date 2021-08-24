import logging
import sys
from json import JSONEncoder, JSONDecoder

from argparse import Namespace

from io import BytesIO

import gc
import threading
import torch
import torch.nn as nn
import os, shutil
import numpy as np
import zipfile

from PIL import Image
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import json
import random
from torchvision import transforms
import IPython


def torch_random_seed(seed):
    """ Strong reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # currently only support this.
        # torch.cuda.set_device(0)
        cuda_device_count = torch.cuda.device_count()
        logging.info(f"GPU number found : {cuda_device_count}")
        torch.cuda.manual_seed(seed)

        if cuda_device_count > 1:
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def new_torch_version():
    if float(torch.__version__[0:3]) < 0.4 :
        return False
    else:
        return True


def get_batch_v03(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def get_batch_vlatest(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    # data = torch.Tensor(source[i:i+seq_len])
    data = Variable(source[i:i + seq_len])
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def to_item_v03(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x
    assert (x.dim() == 1) and (len(x) == 1)
    if isinstance(x, Variable):
        return x.data[0]
    else:
        return x[0]


def to_item_new(x):
    if isinstance(x, (float, int)):
        return x
    else:
        return x.item()


def repackage_hidden_v03(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden_v03(v) for v in h)


def repackage_hidden_new(h):
    if isinstance(h, (Variable, Tensor)):
        return h.detach()
    else:
        return tuple(repackage_hidden_new(v) for v in h)


class DictAttr(object):
    def __init__(self, d):
        self.__dict__ = d

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def update_dict(self, target_dict):
        if isinstance(target_dict, dict):
            for k, v in target_dict.items():
                if k in self.__dict__.keys():
                    self.__dict__[k] = v
                    logging.info(f'DictAttr: change key {k} to {v}')

    def __repr__(self):
        keys = sorted(self.__dict__.keys())
        res = 'Args: { \n'
        for k in keys:
            res += "\t '{}': {},\n".format(k, self.__dict__[k])
        res += '\n}'
        return res

    @classmethod
    def init_from_dict(cls, d):
        a = cls(d)
        for k, v in d.items():
            if isinstance(v, dict):
                a[k] = DictAttr.init_from_dict(v)
        return a


def batchify(data, bsz, args):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_viz_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    print('Experiment dir : {}'.format(path))
    os.mkdir(os.path.join(path, 'visualized_cells'))

    return os.path.join(path, 'visualized_cells')


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    logging.info('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        try:
            os.mkdir(os.path.join(path, 'scripts'))
        except FileExistsError as e:
            logging.warning('Deleting all the previously stored scripts...')
            shutil.rmtree(os.path.join(path, 'scripts'))
            os.mkdir(os.path.join(path, 'scripts'))

        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            if os.path.isdir(script):
                shutil.copytree(script, dst_file, ignore=shutil.ignore_patterns('*.pyc', 'tmp*', '*.ipynb'))
            else:
                shutil.copyfile(script, dst_file)


def save_checkpoint_v2(ckpt, path, backup_weights_epoch=None):
    torch.save(ckpt, path)
    if backup_weights_epoch:
        shutil.copyfile(path, f'{path}.{backup_weights_epoch}')
    

def save_checkpoint(model, optimizer, running_stats, path, scheduler=None, amp=None, finetune=False, backup_weights=False):
    """
    Saving check points for CNNSearchPolicy.
    :param model:
    :param optimizer:
    :param running_stats:
    :param path:
    :param finetune:
    :param backup_weights:
    :return:
    """
    if finetune:
        # m_path = os.path.join(path, 'model.pt')
        # opt_path = os.path.join(path, 'finetune_optimizer.pt')
        prefix = 'finetune_'
    else:
        prefix = ''
    m_path = os.path.join(path, prefix + 'model.pt')
    opt_path = os.path.join(path, prefix + 'optimizer.pt')
    misc_path = os.path.join(path, prefix + 'misc.pt')

    torch.save(model.state_dict(), m_path)
    torch.save(optimizer.state_dict(), opt_path)

    if isinstance(running_stats, int):
        running_stats = {'epoch': running_stats}
    torch.save(running_stats, misc_path)

    epoch = running_stats['epoch']
    if backup_weights:
        for f in [m_path, opt_path, misc_path]:
            shutil.copyfile(f, f + ".{}".format(epoch))

    # save scheduler separately.
    if scheduler:
        s_path = os.path.join(path, prefix + 'scheduler.pt')
        torch.save(scheduler.state_dict(), s_path)
        if backup_weights:
            shutil.copyfile(s_path, s_path + '.{}'.format(epoch))
    if amp:
        s_path = os.path.join(path, prefix + 'amp.pt')
        torch.save(amp.state_dict(), s_path)
        if backup_weights:
            shutil.copyfile(s_path, s_path + '.{}'.format(epoch))
    

def load_checkpoint(path, model=None):
    l_model = torch.load(path)
    if model is not None:
        model.load_state_dict(l_model.state_dict())
    return l_model


def load_checkpoint_v2(path, finetune=False, epoch=None):
    """
    # this will load and return the full checkpoint
    # load the things as follow to resume a training.
    # or for after training.
    New version to keep the old one working.
    :param path:
    :return:
    """
    prefix = 'finetune_' if finetune else ''
    postfix = '.{}'.format(epoch) if epoch else ''
    wrap_name = lambda n: prefix + n + postfix

    load_dict = {}
    # Must load, model., optimizer and misc.pt
    keys = ['model', 'optimizer', 'misc']
    for k in keys:
        p = os.path.join(path, wrap_name(k + '.pt'))
        logging.info(f"loading {k} from {p}.")
        load_dict[k] = torch.load(p)
        # logging.debug(f'dict_keys: {load_dict[k].keys()}')

    s_path = os.path.join(path, wrap_name('scheduler.pt'))
    if os.path.exists(s_path):
        logging.info("loading scheduler from this ")
        load_dict['scheduler'] = torch.load(s_path)
        # logging.debug(f"dict_keys: {load_dict['scheduler'].keys()}")

    a_path = os.path.join(path, wrap_name('amp.pt'))
    if os.path.exists(a_path):
        logging.info("loading amp from this ")
        load_dict['amp'] = torch.load(a_path)

    return load_dict


def save_json(data, filename):
    # class EJSONEncoer(JSONEncoder):
    #     def default(self, o):
    #         try:
    #             if isinstance(o, (Namespace, DictAttr)):
    #                 return o.__dict__
    #             else:
    #                 pass
    #         except:
    #             raise
    def custom_default(o):
        if isinstance(o, (Namespace, DictAttr)):
            return o.__dict__
        else:
            return None

    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True, default=custom_default, indent=4,)


def load_json(filename):
    class CJSONDecoder(JSONDecoder):
        pass

    with open(filename, 'r') as f:
        return json.load(f, parse_int=int, parse_float=float)


def load_args(filename):
    """ load and return the Args """
    a = Namespace()
    a.__dict__ = load_json(filename)
    return a


def load_list(filename, s_type=str):
    with open(filename) as f:
        res = f.readlines()
    res = [s_type(r.rstrip('\n')) for r in res]
    return res


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5, avg_mode=False):
        if not self.training or not dropout:
            return x

        if avg_mode:

            m = x[0].data.new(1, x[0].size(1), x[0].size(2)).bernoulli_(1 - dropout)
            mask = Variable(m.div_(1 - dropout), requires_grad=False)
            mask = mask.expand_as(x[0])
            new_x = torch.stack([mask * out for out in x])

            return new_x

        else:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = Variable(m.div_(1 - dropout), requires_grad=False)
            mask = mask.expand_as(x)

            return mask * x


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.cuda()
    return m


def get_logger(name=__file__, level=logging.INFO, file_handler=None):
    # Only use 1 logger throughtout the entire time!
    logger = logging.getLogger()

    # This is a quite stupid logic if you only use 1 logger.
    # Each time, if a new file handler is passed in, please just flush the existing logger and create a new one.
    if getattr(logger, '_init_done__', None) and file_handler is None:
        logger.info("Init done, return!")
        logger.setLevel(level)
        return logger

    # clean all handler.
    del logger.handlers[:]

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    # Handle the stream handler and file handler if passed.
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)
    if file_handler:
        logger.addHandler(file_handler)

    return logger


def get_file_handler(path):
    # preparation of the logger file
    log_format = '%(asctime)s %(message)s'
    fh = logging.FileHandler(path)
    fh.setLevel(0)
    fh.setFormatter(logging.Formatter(log_format))
    return fh


# Alias definition
# Define the alias for 03 to 04.
if new_torch_version():
    sigmoid = torch.sigmoid
    tanh = torch.tanh
    fn_embedding = lambda _: F.embedding
    to_item = to_item_new
    repackage_hidden = repackage_hidden_new
    get_batch = get_batch_vlatest
    clip_grad_norm = torch.nn.utils.clip_grad_norm_
else:
    sigmoid = F.sigmoid
    tanh = F.tanh
    fn_embedding = lambda embed: embed._backend.Embedding.apply
    to_item = to_item_v03
    repackage_hidden = repackage_hidden_v03
    get_batch = get_batch_v03
    clip_grad_norm = torch.nn.utils.clip_grad_norm


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    # print(masked_embed_weight)
    # print(words.max()
    #       )
    # print(embed)
    X = fn_embedding(embed)(words, masked_embed_weight,
                            padding_idx, embed.max_norm, embed.norm_type,
                            embed.scale_grad_by_freq, embed.sparse
                            )
    return X


def count_parameters_in_MB_search(model, arch, verbose=False):
    """ Only for refinenet kind of architecture... 
    """
    def count_params(any_model):
        return np.sum(np.prod(v.size()) for name, v in any_model.named_parameters())/1e6

    # search module.
    # for the architecture, we do the following
    total_params = 0
    for i in range(1,5):
        block = getattr(model.scratch, f'refinenet{i}')
        for k, c in arch.items():
            if f'fusion{i}' in k:
                edge_c = k.split('_')[1]
                if 'upsample' in edge_c:
                    num_p = count_params(block.upsample[c])
                else:
                    num_p = count_params(block.edges[edge_c][c])
                if verbose:
                    logging.info(f'for fusion layer {i} (i.e. refinenet{i}) edge {edge_c}: '
                          f'choice {c}, num_params {num_p}')
                total_params += num_p
    
    return total_params



def count_parameters_in_MB(model):
    
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def dumpclean(obj, print_fn=print):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print_fn(k)
                dumpclean(v)
            else:
                print_fn('%s : %s' % (k, v))
    elif isinstance(obj, list):
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print_fn(v)
    else:
        print_fn(obj)


# FOR CNN

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(cutout_size):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def debug_data_transforms_cifar10_like_imagenet(cutout_size):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


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


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, num_workers=1, target_transform=None):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
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
                        end_index = num_files if i == num_workers - 1 else num_per_worker * (i + 1)
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


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None,target_transform=None):
        super(ZipDataset, self).__init__() 
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        with zipfile.ZipFile(self.path, 'r') as reader:
            classes, class_to_idx = self.find_classes(reader)
            fnames = sorted(reader.namelist())
        for fname in fnames:
            if self.is_directory(fname):
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
    def find_classes(reader):
        classes = [ZipDataset.get_target(name) for name in reader.namelist() if ZipDataset.is_directory(name)]
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


class InMemoryZipDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, num_workers=1,target_transform=None):
        super(InMemoryZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
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
                end_index = num_files if i == num_workers - 1 else (i + 1) * num_per_worker
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

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def clone_detach_model_state_dict(model:nn.Module):
    """
    Crucial important, the state is changible.
    :param model:
    :return:
    """
    # return a state dict which is total isolate.
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v.detach().clone()
    return state_dict


def arch_to_genotype(arch, method):
    import nasws.cnn.policy.darts_policy.genotypes as darts_space_genotype
    if method.lower() == 'nasbench':
        if arch == 'random':
            # genotype = random_spec(nasbench=NASBench_v2('/home/yukaiche/pycharm/nasbench/nasbench_only108.tfrecord'))
            genotype = model_spec_demo
        elif arch == 'test-1':
            genotype = test_model_max(1)
        else:
            # Read as 'hash'
            logger.info("Treat the arch as hash: {}".format(arch))
            nasbench_v2 = NASBench_v2(
                './data/nasbench/nasbench_only108.tfrecord', only_hash=False)
            logger.info("Load from NASBench dataset v2")
            display_cell(nasbench_v2.query_hash(arch))
            model_arch = nasbench_v2.hash_to_model_spec(arch)
            logger.info('Is this valid? {}'.format(
                nasbench_v2.is_valid(model_arch)))
            return model_arch
    elif method.lower() == 'nao':
        genotype = darts_space_genotype.nao_final_genotypes_by_seed(int(arch))
        logging.info("Using NAO final searched model under seed {}: Genotype {}".format(
            arch, genotype))
    elif method.lower() == 'darts':
        genotype = darts_space_genotype.darts_final_genotype_by_seed(int(arch))
        logging.info(("Using DARTS search: {}".format(genotype)))
    elif method.lower() == 'enas':
        genotype = darts_space_genotype.enas_final_genotypes_by_seed(int(arch))
        logging.info("Using ENAS final search model under seed {}: Genotype {}".format(
            arch, genotype))
    elif method.lower() == 'random':
        genotype = darts_space_genotype.random_generate_genotype_by_seed(int(arch))
        logging.info("Using guided random final search model under seed {}: Genotype {}".format(
            arch, genotype))
    elif method.lower() == 'spos':
        genotype = darts_space_genotype.spos_generate_genotype_by_seed(int(arch))
        logging.info("Using SPOS final search model under seed {}: Genotype {}".format(
            arch, genotype))
    elif method.lower() == 'rankloss':
        genotype = darts_space_genotype.rankloss_generate_genotype_by_seed(int(arch))
        offset = 0
        genotype = genotype[offset]
        logging.info("Using rankloss final search model under seed {}: Genotype {}".format(
            arch, genotype))
    elif method.lower() == 'nao_nds':
        """ load the arch from string based arch to model spec """
        from nasws.cnn.policy.nao_policy.utils_for_darts import NAOParsingDarts
        model_spec = NAOParsingDarts.parse_arch_to_model_spec(eval(arch))
        genotype = model_spec.to_darts_genotype()
        logging.info("Using nds final search model: \n\tarch {} \n\tGenotype {} ".format(
            arch, genotype))
    elif method.lower() == 'published':
        logging.info(f'Using the published architecture from other papers: {arch}')
        genotype = darts_space_genotype.__dict__[arch]
        logging.info(f'Genotype: {genotype}')
    else:
        raise NotImplementedError
    return genotype


def wrap_config(config, prefix, keys, args):
    logging.info("Wrapping the keys into search_policy_configs")
    for k in keys:
        if prefix == '':
            v = getattr(args, f'{k}')
        else:
            v = getattr(args, f'{prefix}_{k}')
        logging.info(f"Setting config key {k} to {v}")
        setattr(config, k, v)