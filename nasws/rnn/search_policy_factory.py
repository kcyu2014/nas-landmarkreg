# from nasws import RandomSearchPolicy
from .enas_policy import enas_configs
from .enas_policy.enas_policy_search import EnasSearchPolicy
from .darts_policy.darts_search_policy import DartsSearchPolicy
from .darts_policy import darts_search_configs as darts_configs
from .nao_policy import nao_configs as nao_args
from .nao_policy import NaoSearchPolicy
from .random_policy.weight_sharing_ranking import WeightSharingSearch
from .random_policy.weight_sharing_ranking_random_on_batch import WeightSharingRandomRank
from .softws.soft_weight_sharing_ranking_random_on_batch import SoftWeightSharingRandomRank, SoftWeightSharingRandomRankFairNAS
from .softws.softws_random_with_policy import SoftWeightSharingRandomRankWithSearch
from .random_policy.fairnas_weight_sharing import FairSampleWeightSharingRandomRank


class SearchPolicyFactory:

    @staticmethod
    def factory(args):
        if args.search_policy == 'random':
            # return RandomSearchPolicy(args)
            pass
        elif args.search_policy == 'ws':
            return WeightSharingSearch(args)
        elif args.search_policy == 'ws_r_batch':
            return WeightSharingRandomRank(args)
        elif args.search_policy == 'soft_ws_r_batch':
            # support the soft weight sharing
            if args.softws_policy is None:
                return SoftWeightSharingRandomRank(args)
            else:
                # Using policy based Softws Random on batch.
                return SoftWeightSharingRandomRankWithSearch(args)
        elif args.search_policy == 'fairnas':
            if args.softws_num_param_per_node > 1:
                return SoftWeightSharingRandomRankFairNAS(args)
            else:
                return FairSampleWeightSharingRandomRank(args)
        elif args.search_policy == 'darts':
            configs = darts_configs.darts_configs
            configs.seed = args.seed
            configs.main_path = args.main_path
            configs.num_intermediate_nodes = args.num_intermediate_nodes
            configs.concat = args.concat
            configs.gpu = args.gpu
            return DartsSearchPolicy(configs)
        elif args.search_policy == 'nao':
            configs = nao_args.nao_configs
            configs.seed = args.seed
            configs.main_path = args.main_path
            configs.use_avg_leaf = args.use_avg_leaf
            configs.gpu = args.gpu
            configs.top_archs = args.top_archs
            configs.max_new_archs = args.max_new_archs
            configs.random_new_archs = args.random_new_archs
            configs.controller_num_seed_arch = args.controller_num_seed_arch
            configs.controller_train_epochs = args.controller_train_epochs
            configs.child_eval_every_epochs = args.child_eval_every_epochs
            configs.controller_batch_size = args.controller_batch_size
            configs.num_intermediate_nodes = args.num_intermediate_nodes
            # configs.handle_hidden_mode = args.handle_hidden_mode
            # configs.emulate_ws_solutions = args.emulate_ws_solutions
            print('return nao policy with: top_archs {}, random_new_archs {}, max_new_archs{}, '
                  'controller_num_seed_arch {}, controller_batch_size {}'.format(
                configs.top_archs,
                configs.random_new_archs,
                configs.max_new_archs,
                configs.controller_num_seed_arch,
                configs.controller_batch_size,
                configs.random_new_archs,
                configs.random_new_archs,
                configs.random_new_archs,
            ))
            return NaoSearchPolicy(configs)
        elif 'enas' in args.search_policy:
            if 'small' in args.search_policy:
                configs = enas_configs.enas_config_small
            else:
                configs = enas_configs.enas_config
            configs.seed = args.seed
            configs.gpu = args.gpu
            configs.main_path = args.main_path
            configs.num_blocks = args.num_intermediate_nodes
            print("If you need to change the config of ENAS, please go to enas_configs/enas_config.py. "
                  "we do not do porting through the universal API here. ")
            return EnasSearchPolicy(configs)
        else:
            raise NotImplementedError
