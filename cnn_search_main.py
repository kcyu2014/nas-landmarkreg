"""
Entry point for CNN Search Main file.
"""
import os
import logging
import traceback, sys, code
import copy
import glob
import pickle
import random
import tensorflow as tf
tf.enable_eager_execution()
import torch
# change the os.environ to use the nds.db

os.environ['NASBENCHMARK_DIR'] = f'{os.getcwd()}/data/nni_nds'
from utils import create_exp_dir, create_dir, DictAttr
from nasws.cnn.policy import cnn_search_configs as configs
from nasws.cnn.policy.cnn_search_policy_factory import CNNCellSearchPolicyFactory
import nasws.cnn.search_space.search_space_utils as search_space


args = configs.parser.parse_args()
print(f">++++ USING PYTORCH VERSION {torch.__version__} ++++++")

# creating the main directory
if args.main_path is None:
    if args.model_spec_hash is not None:
        # evaluate code should be called.
        args.main_path = 'modelspec_{}-nodes_{}-SEED{}_{}-cuda9{}'.format(
            args.model_spec_hash,
            args.num_intermediate_nodes,
            args.seed_range_start,
            args.seed_range_end,
            torch.cuda.get_device_name(0).replace(' ', '-').replace('(', '').replace(')', ''),
        )
    else:
        args.main_path = 'SEED{}_{}-cuda9{}'.format(
            args.seed_range_start,
            args.seed_range_end,
            torch.cuda.get_device_name(0).replace(' ', '-').replace('(', '').replace(')', ''),
        )
    args.main_path = os.path.join(args.test_dir, args.main_path)
    print(">=== Create main path ====")
    # Copying all the path py file into the folder.
    create_exp_dir(args.main_path)
    # create_exp_dir(args.main_path, scripts_to_save=glob.glob('*.py') + glob.glob('nasws'))
else:
    print(">=== using existing main path ====")
print(args.main_path)

# creating the tensorboard directory
tboard_path = os.path.join(args.main_path, args.tboard_dir)
args.tboard_dir = tboard_path
create_dir(tboard_path)
# list for the final dictionary to be used for dataframe analysis
model_spec_ids = []
model_specs = []
validation_acc = []
seeds = []
range_tests = [value for value in range(args.seed_range_start, args.seed_range_end)]
all_evaluations_dicts = {}

backup_args = args
backup_args_dict = DictAttr(vars(args))

for value in range_tests:
    args = copy.deepcopy(args)
    temp_eval_dict = {}
    args.seed = value
    random.seed(args.seed)

    print(f">===== Experiment with SEED {value} =====")
    try:
        if args.model_spec_hash is not None:
            chosen_mspec_id, chosen_mspec = search_space.get_fixed_model_spec(args)
        else:
            # creating the search strategy object from the Search Policy Factory
            search_policy = CNNCellSearchPolicyFactory.factory(args)

            print(">===== Start the search process ... =====")
            # getting the best architecture from the chosen policy
            # return the top-5 architectures.
            chosen_mspec_id, chosen_mspec = search_policy.run()
            print('> chosen ids', chosen_mspec_id)
            print('> chosen specs', chosen_mspec)
            print(">===== Finish search, delete the search policy to release all GPU memory =====")
            # IPython.embed(header='Set search policy to none and check if this sucessfully empty the memory usage')
            # search_policy.cpu()
            # torch.cuda.max_memory_allocated() = 9,218,769,920 (9G)
            # after empty cache, = 1,127,446,528 (1G), but after deleting search_policy, it still remains. WHY?

            if args.plot_loss_landscape:
                print(">===== Start the loss landscape process ... =====")
                # proceed with landscape computation
                from visualization.loss_landscape import SuperNetWrapper, LossLandscapePloter
                net = search_policy.parallel_model
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                
                net = net.cpu()
                if 'nasbench101' in args.search_space:
                    net.unused_modules_back()
                    net.ALWAYS_FULL_PARAMETERS = True
                
                # move to CPU from now on.
                supernet = SuperNetWrapper(
                    supernet=net, 
                    model_specs=search_policy.search_space.topologies, 
                    change_model_spec_fn=search_policy.change_model_spec_fn,
                    # post_fn=lambda m: m.unused_module_back() if 'nasbench101' in args.search_space else None
                    always_same_arch=args.plot_loss_landscape_always_same_arch
                )

                os.makedirs(args.main_path + '/loss-landscape', exist_ok=True)
                ploter = LossLandscapePloter(
                    model=supernet,
                    x=f'-1:1:{args.plot_loss_landscape_step}',
                    y=f'-1:1:{args.plot_loss_landscape_step}',
                    model_file=args.main_path + (f'/loss-landscape/arch-{args.plot_loss_landscape_always_same_arch}' if args.plot_loss_landscape_always_same_arch >= 0 else '/loss-landscape/checkpoint.pt'),
                    plot=True,
                    cuda=True,
                    eval_loss_type='supernet',
                    no_resume=args.plot_loss_landscape_noresume,
                    
                )
                ploter.plot_2d_surface()
                print(">===== Loss landscape process finished !! =====")
            else:
                model_spec_ids.append(chosen_mspec_id)
                model_specs.append(chosen_mspec)

    except Exception as e:
        if args.debug:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
            frame = last_frame().tb_frame
            ns = dict(frame.f_globals)
            ns.update(frame.f_locals)
            code.interact(local=ns)
            # Handle the exception and save all necessary data.
            raise e
        print(e)
        print('ERROR ENCOUNTERED IN THIS SEARCH, MOVING ON NEXT SEARCH')

        rank_dataframe_dict = {'seeds': seeds,
                               'gen_ids': model_spec_ids,
                               'genotypes': model_specs,
                               'last_valid_acc': validation_acc
                               }

        rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                         args.evaluation_seed,
                                                                                                         chosen_mspec_id))
        if args.evaluate_after_search:
            with open(rank_save_path, 'wb') as handle:
                pickle.dump(rank_dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                                   args.evaluation_seed,
                                                                                                                   chosen_mspec_id))

            with open(seeds_save_path, 'wb') as handle:
                pickle.dump(all_evaluations_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        raise e
            # continue

# saving with pickle our dictionaries #
rank_dataframe_dict = {'seeds': seeds,
                       'gen_ids': model_spec_ids,
                       'genotypes': model_specs,
                       'last_valid_acc': validation_acc
                       }

rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                 args.evaluation_seed,
                                                                                                 chosen_mspec_id))
print(rank_dataframe_dict)

if args.evaluate_after_search:
    with open(rank_save_path, 'wb') as handle:
        pickle.dump(rank_dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                           args.evaluation_seed,
                                                                                                           chosen_mspec_id))

    with open(seeds_save_path, 'wb') as handle:
        pickle.dump(all_evaluations_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)





