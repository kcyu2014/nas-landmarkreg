import glob
import time
import torch

from nasws import search_configs as configs
import os
import pickle

from nasws.search_policy_factory import SearchPolicyFactory
from evaluation_phase import EvaluationPhase
from utils import create_exp_dir, create_dir
from genotypes import PRIMITIVES, Genotype, get_forced_genotype_by_name


def get_forced_genotype(forced_geno_id, num_nodes):

    operations = [op for op in PRIMITIVES if op is not 'none']
    num_operations = len(operations)

    gene = [(operations[0], 0) for _ in range(num_nodes)]

    current_div_result = forced_geno_id
    current_node_id = num_nodes - 1

    while current_div_result > 0:
        # print(current_div_result, current_node_id, self.num_operations)
        current_div_result, prec_op_id = divmod(current_div_result, ((current_node_id + 1) * num_operations))

        prec_node_id, operation_id = divmod(prec_op_id, num_operations)

        # updating the edge for the current node slot with the new ids
        gene[current_node_id] = (operations[operation_id], prec_node_id)

        # updating to the next node id of the genotype slot, from bottom to top
        current_node_id -= 1

    return forced_geno_id, Genotype(recurrent=gene, concat=range(num_nodes + 1)[-num_nodes:])


# retrieving the args
args = configs.parser.parse_args()

args_dict = vars(args)

print(f">++++ USING PYTORCH VERSION {torch.__version__} ++++++")

# creating the main directory
if args.main_path is None:
    args.main_path = '{}_nodes_{}_use_leaf_{}_SEED{}-{}_cuda9-{}'.format(
        'geno_id_{}'.format(args.forced_genotype) if args.forced_genotype else 'policy-{}'.format(args.search_policy),
        args.num_intermediate_nodes,
        args.use_avg_leaf,
        args.seed_range_start,
        args.seed_range_end,
        torch.cuda.get_device_name(0).replace(' ', '-').replace('(', '').replace(')', ''),
        # time.strftime("%Y%m%d-%H%M%S")
    )
    args.main_path = os.path.join(args.test_dir, args.main_path)
    print(">=== Create main path ====")
    create_exp_dir(args.main_path, scripts_to_save=glob.glob('*.py') + glob.glob('nasws'))
else:
    print(">=== USING Existing MAIN PATH ====")

print(args.main_path)

# creating the tensorboard directory
tboard_path = os.path.join(args.main_path, args.tboard_dir)
args.tboard_dir = tboard_path
create_dir(tboard_path)

# list for the final dictionary to be used for dataframe analysis
gen_ids = []
genotypes = []
validation_ppl = []
seeds = []

range_tests = [value for value in range(args.seed_range_start, args.seed_range_end)]

all_evaluations_dicts = {}

for value in range_tests:

    temp_eval_dict = {}

    # modifying the configs
    args.seed = value
    print(f">===== Experiment with SEED {value} =====")
    try:
        if args.forced_genotype is not None:
            chosen_gen_id, chosen_genotype = get_forced_genotype(args.forced_genotype, args.num_intermediate_nodes)
        elif args.forced_genotype_by_name is not None:
            chosen_gen_id = 0
            chosen_genotype = get_forced_genotype_by_name(args.forced_genotype_by_name)
        else:
            # creating the search strategy object from the Search Policy Factory
            search_policy = SearchPolicyFactory.factory(args)
            print(">===== Start the search process ... =====")
            # getting the best architecture from the chosen policy
            chosen_gen_id, chosen_genotype = search_policy.run()

        if args.evaluate_after_search:
            print(">===== Start the evaluation process ... =====")
            # initializing the Evaluation routine to evaluate the chosen genotype
            evaluation_phase = EvaluationPhase(args=args,
                                               genotype_id=chosen_gen_id,
                                               genotype=chosen_genotype)

            # training from scratch the chosen genotype
            train_loss_list, train_ppl_list, valid_loss_list, valid_ppl_list, test_loss, test_ppl = evaluation_phase.run()

            temp_eval_dict['train_losses'] = train_loss_list
            temp_eval_dict['train_ppls'] = train_ppl_list
            temp_eval_dict['valid_losses'] = valid_loss_list
            temp_eval_dict['valid_ppls'] = valid_ppl_list
            temp_eval_dict['test_loss'] = test_loss
            temp_eval_dict['test_ppl'] = test_ppl

            # saving the metrics dict in the genotypes dictionary
            all_evaluations_dicts[value] = temp_eval_dict

            # saving infos regarding the ranking of the genotypes
            seeds.append(value)
            gen_ids.append(chosen_gen_id)
            genotypes.append(chosen_genotype.recurrent)
            validation_ppl.append(valid_ppl_list[-1])

    except Exception as e:
        # Handle the exception and save all necessary data.
        print(e)
        print('ERROR ENCOUNTERED IN THIS SEARCH, MOVING ON NEXT SEARCH')

        rank_dataframe_dict = {'seeds': seeds,
                               'gen_ids': gen_ids,
                               'genotypes': genotypes,
                               'last_valid_ppl': validation_ppl
                               }

        rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                         args.evaluation_seed,
                                                                                                         chosen_gen_id))
        if args.evaluate_after_search:
            with open(rank_save_path, 'wb') as handle:
                pickle.dump(rank_dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                             args.evaluation_seed,
                                                                                                             chosen_gen_id))

            with open(seeds_save_path, 'wb') as handle:
                pickle.dump(all_evaluations_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            raise
            # continue

# saving with pickle our dictionaries #
rank_dataframe_dict = {'seeds': seeds,
                       'gen_ids': gen_ids,
                       'genotypes': genotypes,
                       'last_valid_ppl': validation_ppl
                       }

rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                         args.evaluation_seed,
                                                                                                         chosen_gen_id))
print(rank_dataframe_dict)

if args.evaluate_after_search:
    with open(rank_save_path, 'wb') as handle:
        pickle.dump(rank_dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                             args.evaluation_seed,
                                                                                                             chosen_gen_id))

    with open(seeds_save_path, 'wb') as handle:
        pickle.dump(all_evaluations_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)





