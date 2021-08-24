import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--emsize', type=int, default=850,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=850,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=850,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--train_num_batches', type=int, default=18)
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='EXP', help='Path prefix: Train from scratch')
parser.add_argument('--save_every_epoch', type=int, default=50, help='evaluate and save every x epochs.')
parser.add_argument('--main_path', type=str, default=None, help='path to save EXPs.')
parser.add_argument('--resume_path', type=str, default=None, help='path to save the final model')
parser.add_argument('--test_dir', type=str, default='none', help='path to save the tests directories')
parser.add_argument('--tboard_dir', type=str, default='runs', help='path to save the tboard logs')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20, help='max sequence length')
parser.add_argument('--seed_range_start', type=int, default=None)
parser.add_argument('--seed_range_end', type=int, default=None)
parser.add_argument('--gen_ids_start', type=int, default=None)
parser.add_argument('--gen_ids_range', type=int, default=None)
parser.add_argument('--single_gpu', default=True, action='store_false', help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')

# TODO Clean the PSO stuff.
parser.add_argument('--start_using_pso', type=int, default=5)
parser.add_argument('--updates_pso', type=int, default=1)
parser.add_argument('--population_size', type=int, default=25, help='number_of_particles')
parser.add_argument('--handle_hidden_mode', type=str, default='None', choices=['NORM', 'None', 'RELOAD', "ACTIVATION"])
parser.add_argument('--clip_hidden_norm', type=float, default=25., )
parser.add_argument('--use_training_phase', default=True, action='store_false')
parser.add_argument('--visualize', default=False, action='store_false')
parser.add_argument('--num_operations', type=int, default=4, help='valid operations in search space')
parser.add_argument('--num_intermediate_nodes', type=int, default=8)
parser.add_argument('--concat', type=int, default=8)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--w_inertia', type=float, default=0.5)
parser.add_argument('--c_local', type=int, default=1)
parser.add_argument('--c_global', type=int, default=2)
parser.add_argument('--edge_select_mode', type=str, default='softmax')
parser.add_argument('--search_policy', type=str, default='random')
parser.add_argument('--reduce_clusters', default=False, action='store_false')
parser.add_argument('--uniform_genos_init', default=True, action='store_false')
parser.add_argument('--use_pretrained', action='store_true')
parser.add_argument('--use_fixed_slot', action='store_true')
parser.add_argument('--use_random', action='store_true')
parser.add_argument('--random_id_eval', action='store_true')
parser.add_argument('--minimum', type=int, default=None)
parser.add_argument('--range_coeff', type=int, default=2)
parser.add_argument('--use_matrices_on_edge', action='store_true')
parser.add_argument('--use_glorot', default=False, action='store_true')
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--evaluate_after_search', type=str2bool, dest='evaluate_after_search')
parser.add_argument('--evalaute_every_epoch', type=int, default=20)
# parser.set_defaults(evaluate_after_search=True)

parser.add_argument('--genotype_id', type=int, default=None)
parser.add_argument('--forced_genotype', type=int, default=None)
parser.add_argument('--forced_genotype_by_name', type=str, default=None)
parser.add_argument('--evaluation_seed', type=int, default=1278)
parser.add_argument('--use_avg_leaf', default=False, action='store_true')

## NAO SEARCH CONFIGS ##
parser.add_argument('--top_archs', type=int, default=None)
parser.add_argument('--random_new_archs', type=int, default=None)
parser.add_argument('--max_new_archs', type=int, default=None)
parser.add_argument('--controller_num_seed_arch', type=int, default=None)
parser.add_argument('--controller_train_epochs', type=int, default=None)
parser.add_argument('--child_eval_every_epochs', type=int, default=None)
parser.add_argument('--controller_batch_size', type=int, default=None)
parser.add_argument('--emulate_ws_solutions', type=int, default=None)
parser.add_argument('--genotype_start', type=int, default=None)
parser.add_argument('--genotype_end', type=int, default=None)

# Soft-weight-sharing nodes
parser.add_argument('--softws_num_param_per_node', type=int, default=1, help='Number of node per layer. '
                                                                             'only support int for now')
parser.add_argument('--genotype_mapping_fn', type=str, default=None, help='Specify node 2 case test with '
                                                                          'soft-weightsharing. map_v1, map_v2 ...')
parser.add_argument('--softws_policy', type=str, default=None, help='SoftWS policy here.')
parser.add_argument('--softws_lr', type=float, default=3e-3, help='Softws learning rate.')
parser.add_argument('--softws_init_v', type=float, default=0.1, help='Initial value for soft ws.')
parser.add_argument('--softws_wdecay', type=float, default=0.001, help='Softws weight decay.')
parser.add_argument('--unrolled', default=False, action='store_true')