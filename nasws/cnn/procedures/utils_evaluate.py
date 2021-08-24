"""
Evaluate procedure utils.


Mainly support the sampling methods here there.

"""
import logging
import random
from nasws.cnn.search_space.api import CNNSearchSpace


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def guided_random_sampler():
    pass


def _eval_step_(_id, spec, query_fn, res_dict, best_valids, best_tests, prefix=None):
    data = query_fn(spec)
    res_dict[_id] = data
    # It's important to select models only based on validation accuracy, test
    # accuracy is used only for comparing different search trajectories.
    if data['validation_accuracy'] > best_valids[-1]:
        best_valids.append(data['validation_accuracy'])
        best_tests.append(data['test_accuracy'])
    else:
        best_valids.append(best_valids[-1])
        best_tests.append(best_tests[-1])
    logging_step(prefix, _id, data['validation_accuracy'], data['test_accuracy'])
    return res_dict, best_valids, best_tests


def logging_step(prefix, mid, val, test):
    logging.info(f"{prefix} model {mid}: {val} {test}.")


def run_random_search_over_cnn_search(search_space: CNNSearchSpace, args, query_fn):
    # return a model spec
    seen_model, best_valids, best_tests = 0, [0.0], [0.0]
    res_dict = {}
    logging.info("Begin random search over CNN Search Space. {}".format(search_space))
    res_specs = {}
    while seen_model < args.evaluate_step_budget:
        _id, spec = search_space.random_topology_random_nas()
        # IPython.embed()
        res_dict, best_valids, best_tests = _eval_step_(_id, spec, query_fn, res_dict, best_valids, best_tests,
                                                        prefix='random search')
        seen_model += 1
        res_specs[_id] = spec

    # best_valids and best_tests are used to plot the comparison curves
    return res_dict, best_valids, best_tests, res_specs


def run_search_over_fixed_set(spec_dict, query_fn):
    seen_model, best_valids, best_tests = 0, [0.0], [0.0]
    res_dict = {}
    for _id, spec in spec_dict.items():
        res_dict, best_valids, best_tests = _eval_step_(_id, spec, query_fn, res_dict, best_valids, best_tests,
                                                        prefix='fixed set')
        seen_model += 1

    # best_valids and best_tests are used to plot the comparison curves
    return res_dict, best_valids, best_tests


def run_evolutionary_search_on_search_space(search_space:CNNSearchSpace, args, query_fn, population=None):
    """
    Should also define the evolutionary steps.
    :param search_space:
    :param args
    :param candidate_dict:
    :param query_fn: function that return the data point (contains validation_accuracy, and test_accuracy)
    :return:
    """

    seen_model, best_valids, best_tests = 0, [0.0], [0.0]
    population = population or []  # (validation, spec) tuples

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    population_size = args.evaluate_evolutionary_population_size
    res_dict = {}
    res_specs = {}
    # initialize the population size
    for _ in range(population_size):
        s_id, spec = search_space.random_topology()
        seen_model += 1
        res_dict, best_valids, best_tests = _eval_step_(s_id, spec, query_fn, res_dict, best_valids, best_tests,
                                                        prefix='evo random')
        # population[s_id] = spec
        population.append((res_dict[s_id]['validation_accuracy'], spec))
        res_specs[s_id] = spec

    # After the population is seeded, proceed with evolving the population.
    while seen_model < args.evaluate_step_budget:
        sample = random_combination(population, args.evaluate_evolutionary_tournament_size)
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        s_id, spec = search_space.mutate_topology(best_spec, args.evaluate_evolutionary_mutation_rate)
        
        if s_id is None:
            logging.info(f'Found a new architecture! {spec}')
            s_id = spec.hash_spec()
        
        seen_model += 1
        res_dict, best_valids, best_tests = _eval_step_(s_id, spec, query_fn, res_dict, best_valids, best_tests,
                                                        prefix='evo mutate')

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((res_dict[s_id]['validation_accuracy'], spec))
        population.pop(0)
        res_specs[s_id] = spec

    return res_dict, best_valids, best_tests, population, res_specs


def iclr_recent_submodel_sampler(search_space):
    """
    Representing the recent baseline, do if if you have time.
    :param search_space:
    :return:
    """
    pass
