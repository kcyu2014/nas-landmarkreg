import os
import numpy as np
import torch
import shutil
from torch.autograd import variable
from torchvision.transforms import transforms


class cutout(object):
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


def _data_transforms_cifar10(args):
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.compose([
        transforms.randomcrop(32, padding=4),
        transforms.randomhorizontalflip(),
        transforms.totensor(),
        transforms.normalize(cifar_mean, cifar_std),
    ])
    if args.cutout:
        train_transform.transforms.append(cutout(args.cutout_length))

    valid_transform = transforms.compose([
        transforms.totensor(),
        transforms.normalize(cifar_mean, cifar_std),
    ])
    return train_transform, valid_transform


def count_parameters_in_mb(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best ,save_dir):
    filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best is not None:
        best_filename = os.path.join(save_dir, 'model-best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob, inplace=False):
    """ updated to resolve Inplace error bug """
    if drop_prob > 0.:
        keep_prob = 1.- drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).cuda()
        if inplace:
            x.div_(keep_prob)
            x.mul_(mask)
        else:
            x = x / keep_prob * mask
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

