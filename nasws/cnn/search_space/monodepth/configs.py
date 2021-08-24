"""NAS Parser

"""

#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/27 下午2:32
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser("NAS parser. ")

# common space
parser.add_argument('--dataset', type=str, default='cifar10', help="dataset to use.")
parser.add_argument('--num_intermediate_nodes', type=int, default=2, help='NASbench cell vertex')
parser.add_argument('--search_policy', type=str, default='random', help="Search Policy")
parser.add_argument('--search_space', type=str, default='nasbench', help="Search Space to use")
parser.add_argument('--seed_range_start', type=int, default=1268)
parser.add_argument('--seed_range_end', type=int, default=1269)
parser.add_argument('--evaluation_seed', type=int, default=1278, help='default evaluation seeds.')
parser.add_argument('--supernet_train_method', type=str, default='darts', help='Training method for supernet model')
parser.add_argument('--mutator', type=str, default='nonzero', help='Mutator class')
parser.add_argument('--visualize', default=False, action='store_false')
parser.add_argument('--evaluate_after_search', default=False, action='store_true')
parser.add_argument('--tensorboard', default=True, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
# parser.add_argument('--no-tensorbard', dest='tensorboard', action='store_false')
# parser.set_defaults(tensorboard=True)

# search_main configs
parser.add_argument('--save_every_epoch', type=int, default=30, help='evaluate and save every x epochs.')
parser.add_argument('--extensive_save', type=str2bool, default=True,
                    help='Extensive evaluate 3 consequtive epochs for a stable results.')
parser.add_argument('--resume_path', type=str, default=None, help='path to save the final model')
parser.add_argument('--test_dir', type=str, default='none', help='path to save the tests directories')
parser.add_argument('--tboard_dir', type=str, default='runs', help='path to save the tboard logs')

# DARTS based configs.
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--n_worker', type=int, default=10, help='nb worker')
parser.add_argument('--world_size', type=int, default=1, help='world size (for distributed training)')
parser.add_argument('--evaluate_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate_scheduler', type=str, default='cosine', help='learning rate scheduler setting.')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--backbone_learning_rate', type=float, default=0.00001, help='init learning rate for the backbone network')
parser.add_argument('--backbone_weights', type=str, default=None, help='Backbone weights')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpus', type=int, default=1, help='GPU count')
parser.add_argument('--epochs', type=int, default=151, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# For customization customization for nasbench...
parser.add_argument("--model_spec_hash", type=str, default=None, help='Model spec hashing for the NASBench. '
                                                                      'OR index for NAO-DARTS search space.')
parser.add_argument('--nasbench_module_vertices', type=int, default=5, help='Maximum number of vertex.')
# Settings for MixedVertex Op in nas_bench.operations 
parser.add_argument('--nasbenchnet_vertex_type', type=str, default='mixedvertex')
parser.add_argument('--dartsbenchnet_cell_type', type=str, default='op_on_node')
parser.add_argument('--channel_dropout_method', type=str, default='fixed_chunck',
                    help='Channel dropout configs. Used in '
                         'nas_bench.operations.ChannelDropout')
parser.add_argument('--channel_dropout_dropouto', type=float, default=0.0,
                    help='additional dropout operation after the Channel dropout'
                    )
parser.add_argument('--dynamic_conv_method', type=str, default='fixed_chunck',
                    help='Dynamic conv is channel-droupout operate on the convolutional kernels.'
                         'All methods are the same as channel dropout method.')
parser.add_argument('--dynamic_conv_dropoutw', type=float, default=0.0,
                    help='additional kernel dropout.')
parser.add_argument('--path_dropout_rate', type=float, default=0.0, help='NAO way of dropout path, scheduled with the layer id.')
# parser.add_argument('--conv_dropout_rate', type=float, default=0.0, help='dropout the convolutional feature')                  
parser.add_argument('--global_dropout_rate', type=float, default=0.0, help='dropout the final global feature')                  
parser.add_argument('--ofa_max_kernel_size', type=int, default=3, help='OFA max kernel size for validation.')
parser.add_argument('--ofa_parameter_transform', type=int, default=1, help='OFA transform')


## Batchnorm setting here.
parser.add_argument('--bn_type', type=str, default='bn', help="nasbench enable bn during training or not.", choices=['bn', 'wsbn'])
parser.add_argument('--wsbn_train', type=str2bool, default=False, help="nasbench enable bn during training or not.")
parser.add_argument('--wsbn_sync', type=str2bool, default=False, help="Enable Sync BN in NASBench MixedOp.")
parser.add_argument('--wsbn_track_stat', type=str2bool, default=False, help="Disable BN tracking status.")
parser.add_argument('--wsbn_affine', type=str2bool, default=True, help="Diasble BN affine.")
parser.add_argument('--num_archs_subspace', type=int, default=10, help='Number of sub-space architectures to sample.')

## Nips rebuttal
parser.add_argument('--experiment_setting', type=str, default='normal')
parser.add_argument('--controller_random_arch', type=int, default=20)


## MAML experiment
parser.add_argument('--maml_num_train_updates', type=int, default=1, help='step for updating task gradient')
parser.add_argument('--maml_num_inner_tasks', type=int, default=8, help='inner update step to aggregate') # n = 2 after tuning with normal.
parser.add_argument('--maml_task_lr', type=float, default=0.001, help='lr to udate') # this is done by MAML plus
parser.add_argument('--maml_task_optim', type=str, default='sgd', help='Optimizer to update task step') # TODO tune this?
parser.add_argument('--maml_second_order', type=str2bool, default='False', help='use second order')
parser.add_argument('--mamlplus_enable_inner_loop_optimizable_bn_params', type=str2bool,
                    default='False', help='enable inner loop topimizing bn params') # TODO this logic.
parser.add_argument('--mamlplus_number_of_training_steps_per_iter', type=int, default=2,
                    help='steps for each inner loop task.') # n = 2 after tuning with normal.
parser.add_argument('--mamlplus_multi_step_loss_num_epochs', type=int, default=15,
                    help='multi step loss num of epochs.') # TODO alter this.
parser.add_argument('--mamlplus_first_order_to_second_order_epoch', default=50, type=int)
parser.add_argument('--mamlplus_use_multi_step_loss_optimization', default=True, type=str2bool)
parser.add_argument('--mamlplus_dynamic_lr_relu', default=False, type=str2bool)
parser.add_argument('--mamlplus_dynamic_lr_min', default=0.0001, type=float)


## Landmark regularization of the model
parser.add_argument('--landmark_loss_procedure', type=str, default='pairwise_loss',
                    help='Defines the procedure of regularizer of landmark loss.')
parser.add_argument('--landmark_loss_fn', type=str, default='mae_relu',
                    help='Defines the distance function inplaced.')
parser.add_argument('--landmark_num_archs', type=int, default=0, help="Number of landmark architectures you picked.")
parser.add_argument('--landmark_sample_method', type=str, default='random', help="Sampling method for landmarks")
parser.add_argument('--landmark_loss_coef', type=float, default=1., help="Coefficient for the ranking term.")
parser.add_argument('--landmark_loss_coef_scheduler', type=str, default='constant', help="Scheduler for such term")
parser.add_argument('--landmark_loss_weighted', type=str, default='default', help="Use weighted loss in this one.", choices=['default', 'embed', 'infinite'])
parser.add_argument('--landmark_use_valid', type=str2bool, default=False, help="Use a separate dataset to do ranking update.")
parser.add_argument('--landmark_loss_adjacent', type=str2bool, default=False, help="Adjacent computation.")
parser.add_argument('--landmark_loss_adjacent_step', type=int, default=1, help="Adjacent step")
parser.add_argument('--landmark_loss_random_pairs', type=int, default=1, help="random pair loss pairs to train.")
parser.add_argument('--landmark_warmup_epoch', default=0, type=int, help='warmup epochs for this loss function.')

## Subspace id
parser.add_argument('--nasbench_search_space_ws_influence_full', type=str2bool, default=False,
                    help='True to permutate across the entire space, false to permutate the last node only.')

parser.add_argument('--nasbench101_fix_last_channel_config', type=int, default=0, 
                    help='Indicates the last layer mutation id let us consider the current implement now.')
parser.add_argument('--nasbench101_fix_last_channel_sum', type=int, default=2, 
                    help='Indicates the last layer mutation id let us consider the current implement now.')
parser.add_argument('--nasbench201_use_valid_only', type=str2bool, default=False, help='set to True, only enable the valid ones.')
parser.add_argument('--nasbench201_use_isomorphic', type=str2bool, default=False, help='set to True, use only isomorphic architecture.')
parser.add_argument('--nasbench201_use_isomorphic_consider_zeros', type=str2bool, default=False, help='consider only zeros')

## Testing, NAO approach.
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.9)
parser.add_argument('--use_auxiliary_in_search', type=str2bool, default=False)


## New evaluation framework
parser.add_argument('--neweval_num_train_batches', type=int, default=0,
                    help='Number of batches to train before executing evaluation')
parser.add_argument('--neweval_num_train_epoch', type=int, default=0,
                    help='Number of batches to train before executing evaluation')
parser.add_argument('--evaluate_sampler', type=str, default='random', help='Evaluation phase sampler')
parser.add_argument('--evaluate_nb_batch', type=int, default=20, help='Evaluation budget in num of seen model')
parser.add_argument('--evaluate_step_budget', type=int, default=100, help='Evaluation budget in num of seen model')
parser.add_argument('--evaluate_evolutionary_population_size', type=int, default=50, help='Evolutionary sampler size')
parser.add_argument('--evaluate_evolutionary_tournament_size', type=int, default=10, help='Evolutionary tournament size')
parser.add_argument('--evaluate_evolutionary_mutation_rate', type=float, default=1., help='Evolutionary mutation rate')

# Path sampling techniques.
parser.add_argument("--path_sample_method", type=str, default='default', help='Path Sampling Tricks.')


## Apex and Dali related configs, temporaryly we only have these two switch, we follow other mature pipeline settings. 
parser.add_argument("--apex_enable", type=str2bool, default=False, help='This enable the apex package')
parser.add_argument("--dali_enable", type=str2bool, default=False, help='This enable the dali package')


def build_default_args(parse_str=''):
    default_args = parser.parse_args(parse_str)
    return default_args
