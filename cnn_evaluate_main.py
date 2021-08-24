"""Evaluate the search phase of CNN
TO use this, please follow the instructions.

- Given the folder.
- Load the args from the args.json.

- Initialize the 
"""


import copy
import glob
import os
import pickle
import random
import json

import torch
from utils import DictAttr
from nasws.cnn.policy import cnn_search_configs as configs
from nasws.cnn.policy.cnn_search_policy_factory import CNNCellSearchPolicyFactory
import nasws.cnn.search_space.search_space_utils as search_space

def main():
    args = configs.parser.parse_args()
    print(f">++++ USING PYTORCH VERSION {torch.__version__} ++++++")
    # load the args from given folder.
    print("Post search phase from " + args.main_path)
    args_path = glob.glob(args.main_path + "/*/args.json")[0]
    print("Loading args.json from" + args_path)
    with open(args_path) as f:
        l_args = json.load(f)
    dargs = DictAttr(args.__dict__)
    dargs.update_dict(l_args)
    args.__dict__ = dargs.__dict__
    # print("Loaded args:")
    # print(args)

    # creating the tensorboard directory
    model_spec_ids = []
    model_specs = []
    validation_acc = []
    seeds = []
    range_tests = [value for value in range(
        args.seed_range_start, args.seed_range_end)]
    all_evaluations_dicts = {}

    backup_args = args
    backup_args_dict = DictAttr(vars(args))

    for value in range_tests:
        args = copy.deepcopy(args)
        temp_eval_dict = {}
        args.seed = value
        random.seed(args.seed)

        print(f">===== Post Search Experiment with SEED {value} =====")
        try:
            if args.model_spec_hash is not None:
                chosen_mspec_id, chosen_mspec = search_space.get_fixed_model_spec(
                    args)
            else:
                # creating the search strategy object from the Search Policy Factory
                search_policy = CNNCellSearchPolicyFactory.factory(args)
                print(">===== Start the post search process evaluation ... =====")
                search_policy.initialize_model()
                _, _, te, _ = search_policy.initialize_run()
                fitness_dict = search_policy.evaluate(search_policy.epoch, te)
                search_policy.save_results(search_policy.epoch)
                print('> fitness dict', fitness_dict)
                chosen_mspec_id, chosen_mspec = search_policy.post_search_phase()
                print('> chosen ids', chosen_mspec_id)
                print('> chosen specs', chosen_mspec)
                print(">===== Finish Evaluate =====")

            model_spec_ids.append(chosen_mspec_id)
            model_specs.append(chosen_mspec)

        except Exception as e:
            # Handle the exception and save all necessary data.
            print(e)
            print('ERROR ENCOUNTERED IN THIS SEARCH, MOVING ON NEXT SEARCH')

            rank_dataframe_dict = {'seeds': seeds,
                                'gen_ids': model_spec_ids,
                                'genotypes': model_specs,
                                'last_valid_acc': validation_acc
                                }

            rank_save_path = os.path.join(args.main_path,
                                        'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(
                                            args.seed_range_start,
                                            args.evaluation_seed,
                                            chosen_mspec_id))
            if args.evaluate_after_search:
                with open(rank_save_path, 'wb') as handle:
                    pickle.dump(rank_dataframe_dict, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

                seeds_save_path = os.path.join(args.main_path,
                                            'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(
                                                args.seed_range_start,
                                                args.evaluation_seed,
                                                chosen_mspec_id))

                with open(seeds_save_path, 'wb') as handle:
                    pickle.dump(all_evaluations_dicts, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                raise
                # continue

    # saving with pickle our dictionaries #
    rank_dataframe_dict = {'seeds': seeds,
                        'gen_ids': model_spec_ids,
                        'genotypes': model_specs,
                        'last_valid_acc': validation_acc
                        }

    rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(
        args.seed_range_start,
        args.evaluation_seed,
        chosen_mspec_id))
    print(rank_dataframe_dict)

    if args.evaluate_after_search:
        with open(rank_save_path, 'wb') as handle:
            pickle.dump(rank_dataframe_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(
            args.seed_range_start,
            args.evaluation_seed,
            chosen_mspec_id))

        with open(seeds_save_path, 'wb') as handle:
            pickle.dump(all_evaluations_dicts, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()