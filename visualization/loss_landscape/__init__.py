# this module forked from Visualize Loss Landscape paper official repo, and adapted for this project.
import logging
import random
import torch
import argparse
import copy
import torch
import socket
import numpy as np
import torchvision
import torch.nn as nn

from . import mpi4pytorch as mpi
from . import dataloader
from . import evaluation
from . import projection as proj
from . import net_plotter
from . import plot_2D
from . import plot_1D
from . import model_loader
from . import scheduler
from . import plot_surface


def construct_plot_args():
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--no_resume', action='store_true', help='not resume the training')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--eval_loss_fn', default='eval_loss', type=str, help='evaluation loss type')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')

    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='filter', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args('')
    return args


class SuperNetWrapper(nn.Module):

    def __init__(self, supernet, model_specs, change_model_spec_fn, post_fn=None, always_same_arch=None) -> None:
        super(SuperNetWrapper, self).__init__()
        self.supernet = supernet
        self.model_specs = model_specs
        self.change_model_spec_fn = change_model_spec_fn
        self.post_fn = post_fn
        self.always_same_arch = always_same_arch

    def reset(self):
        # random mutation for now
        if self.always_same_arch >=0:
            self.supernet = self.change_model_spec_fn(self.supernet, self.model_specs[self.always_same_arch])
        else:
            self.supernet = self.change_model_spec_fn(self.supernet, random.choice(self.model_specs))

        if self.post_fn:
            self.supernet = self.post_fn(self.supernet)

    def forward(self, inputs):
        return self.supernet(inputs)


class LossLandscapePloter:

    def __init__(self, model, **kwargs) -> None:
        args = construct_plot_args()
        args.__dict__.update(kwargs)
        self.args = args
        self.model = model
    
    def plot_2d_surface(self):
        args = self.args
        
        torch.manual_seed(0)
        #--------------------------------------------------------------------------
        # Environment setup
        #--------------------------------------------------------------------------
        if args.mpi:
            comm = mpi.setup_MPI()
            rank, nproc = comm.Get_rank(), comm.Get_size()
        else:
            comm, rank, nproc = None, 0, 1

        # in case of multiple GPUs per node, set the GPU to use for each rank
        if args.cuda:
            if not torch.cuda.is_available():
                raise Exception('User selected cuda option, but cuda is not available on this machine')
            gpu_count = torch.cuda.device_count()
            torch.cuda.set_device(rank % gpu_count)
            logging.info('Rank %d use GPU %d of %d GPUs on %s' %
                 (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

        #--------------------------------------------------------------------------
        # Check plotting resolution
        #--------------------------------------------------------------------------
        try:
            args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
            args.xnum = int(args.xnum)
            args.ymin, args.ymax, args.ynum = (None, None, None)
            if args.y:
                args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                args.ynum = int(args.ynum)
                assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

        #--------------------------------------------------------------------------
        # Load models and extract parameters
        #--------------------------------------------------------------------------
        net = self.model
        w = net_plotter.get_weights(net) # initial parameters
        s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
        if args.ngpu > 1:
            # data parallel with multiple GPUs on a single node
            net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        #--------------------------------------------------------------------------
        # Setup the direction file and the surface file
        #--------------------------------------------------------------------------
        dir_file = net_plotter.name_direction_file(args) # name the direction file
        if rank == 0:
            net_plotter.setup_direction(args, dir_file, net)

        surf_file = plot_surface.name_surface_file(args, dir_file)
        if rank == 0:
            plot_surface.setup_surface_file(args, surf_file, dir_file)

        # wait until master has setup the direction file and surface file
        mpi.barrier(comm)

        # load directions
        d = net_plotter.load_directions(dir_file)
        # calculate the consine similarity of the two directions
        if len(d) == 2 and rank == 0:
            similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
            print('cosine similarity between x-axis and y-axis: %f' % similarity)

        #--------------------------------------------------------------------------
        # Setup dataloader
        #--------------------------------------------------------------------------
        # download CIFAR10 if it does not exit
        if rank == 0 and args.dataset == 'cifar10':
            torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

        mpi.barrier(comm)

        trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
                                    args.batch_size, args.threads, args.raw_data,
                                    args.data_split, args.split_idx,
                                    args.trainloader, args.testloader)

        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------
        plot_surface.crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args)
        # plot_surface.crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

        #--------------------------------------------------------------------------
        # Plot figures
        #--------------------------------------------------------------------------
        if args.plot and rank == 0:
            if args.y and args.proj_file:
                plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
            elif args.y:
                plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
            else:
                plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
