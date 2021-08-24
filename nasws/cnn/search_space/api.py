"""
Design API for Isolating CNN search space. It contains all related item to search space.

I always keep one thing, which is, "As simple as possible!!!"

- SearchSpace       (Like database, to store the all possible Topologys)
- ModelSearch       (Network to be trained, when take topology as input, can change to different architectures. )
- ModelSpec          (similar to ModelSpec)

"""
# This is a general model specs, can capture all possible topology architectures
# use this across search space

import numpy as np
import copy
import json
import os
import random
import time
import utils

from .search_space import *
from .ws_vertex import MixedVertex


# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
class BenchmarkDatasetTemplate(object):
    """User-facing API for accessing the NASBench dataset."""

    # Store the model specs with hashs.
    # hash -> model spec
    hash_dict = {}

    # Stores the fixed statistics that are independent of evaluation (i.e.,
    # adjacency matrix, operations, and number of parameters).
    # hash --> metric name --> scalar
    fixed_statistics = {}

    # Stores the statistics that are computed via training and evaluating the
    # model on CIFAR-10. Statistics are computed for multiple repeats of each
    # model at each max epoch length.
    # hash --> epochs --> repeat index --> metric name --> scalar
    computed_statistics = {}

    # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
    # {108} for the smaller dataset with only the 108 epochs.
    valid_epochs = list()

    total_epochs_spent = 0

    def __init__(self, dataset_file, seed=None, config=None):
        """Initialize dataset, this should only be done once per experiment.

        Args:
          dataset_file: path to .tfrecord file containing the dataset.
          seed: random seed used for sampling queried models. Two NASBench objects
            created with the same seed will return the same data points when queried
            with the same models in the same order. By default, the seed is randomly
            generated.
        """
        self.config = config
        random.seed(seed)

        print('Loading dataset from file... This may take a few minutes...')
        start = time.time()
        self.load_dataset_file(dataset_file)
        elapsed = time.time() - start
        print('Loaded dataset in %d seconds' % elapsed)
        self.history = {}
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def train_and_evaluate(self, model_spec, config, model_dir):
        """
        :param model_spec:
        :param config:
        :param model_dir:
        :return: metadata. check self.evaluate for more information.
        """
        raise NotImplementedError("Should be done by subclass.")

    def preprocess_dataset_from_given_files(self):
        """ process raw data. """
        raise NotImplementedError("Should be don in subclass.")

    def create_model_spec(self, fixed_data):
        """
        Used in load_dataset_file, map from fixed_data into ModelSpec accordingly.
        :param fixed_data: self.fixed_statistics[key]
        :return: ModelSpec
        """
        raise NotImplementedError("create the model spec accordingly. done in subclass, used in load_dataset_file")

    def load_dataset_file(self, dataset_file):
        """load the preprocessed dataset."""
        try:
            d = utils.load_json(dataset_file)
            self.fixed_statistics = d['fixed_stat']
            # preload the dataset and transform 'epoch' into int
            compute_stat_dict = d['compute_stat']
            new_dict = {}
            for k, entry in compute_stat_dict.items():
                new_dict[k] = {int(epoch): v for epoch, v in entry.items()}
            self.computed_statistics = new_dict
            self.hash_dict = {h: self.create_model_spec(fix_stat)
                              for h, fix_stat in self.fixed_statistics.items()}
        except FileNotFoundError:
            print("File not found, creating the dataset by loading from the disk.")
            self.preprocess_dataset_from_given_files()
            self.save_dataset_file()

    def save_dataset_file(self):
        save_data = {'fixed_stat': self.fixed_statistics, 'compute_stat': self.computed_statistics}
        utils.save_json(save_data, self.dataset_file)

    def _perf_fn(self, data):
        raise NotImplementedError("this should be done in subclass")

    def _loss_fn(self, data):
        raise NotImplementedError("_loss_fn should be overrided.")

    def query_perf(self, model_spec):
        """ query the performance """
        return self._perf_fn(self.query(model_spec))

    def query_loss(self, model_spec):
        """ query the loss """
        return self._loss_fn(self.query(model_spec))

    def query(self, model_spec, epochs=None, stop_halfway=False):
        """
        TODO make this the default one, to reduce excesive amount of hard code.
        :param model_spec:
        :param epochs:
        :param stop_halfway:
        :return:
        """
        epochs = epochs or self.valid_epochs[-1]
        if epochs not in self.valid_epochs:
            raise ValueError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)

        query_res = {}
        for k in self.fixed_statistics_keys:
            query_res[k] = fixed_stat[k]

        # use average results.
        num_repeats = len(computed_stat[epochs])
        average_compute_data = {}
        for run_id, data in computed_stat[epochs].items():
            for k in data.keys():
                average_compute_data[k] = data[k] / num_repeats if k not in average_compute_data.keys() else \
                    average_compute_data[k] + data[k] / num_repeats
                average_compute_data[f'{k}_run-{run_id}'] = data[k]

        for k in self.computed_statistics_keys:
            if k in average_compute_data.keys():
                query_res[k] = average_compute_data[k]

        if stop_halfway:
            self.total_epochs_spent += int(epochs) // 2
        else:
            self.total_epochs_spent += int(epochs)

        return query_res

    def is_valid(self, model_spec):
        """Checks the validity of the model_spec.

        For the purposes of benchmarking, this does not increment the budget
        counters.

        Args:
          model_spec: ModelSpec object.

        Returns:
          True if model is within space.
        """
        try:
            self._check_spec(model_spec)
        except Exception:
            return False

        return True

    def get_budget_counters(self):
        """Returns the time and budget counters."""
        return self.training_time_spent, self.total_epochs_spent

    def reset_budget_counters(self):
        """Reset the time and epoch budget counters."""
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def evaluate(self, model_spec, model_dir):
        """Trains and evaluates a model spec from scratch (does not query dataset).

        This function runs the same procedure that was used to generate each
        evaluation in the dataset.  Because we are not querying the generated
        dataset of trained models, there are no limitations on number of vertices,
        edges, operations, or epochs. Note that the results will not exactly match
        the dataset due to randomness. By default, this uses TPUs for evaluation but
        CPU/GPU can be used by setting --use_tpu=false (GPU will require installing
        tensorflow-gpu).

        Args:
          model_spec: ModelSpec object.
          model_dir: directory to store the checkpoints, summaries, and logs.

        Returns:
          dict contained the evaluated data for this object, same structure as
          returned by query().
        """
        # Metadata contains additional metrics that aren't reported normally.
        # However, these are stored in the JSON file at the model_dir.
        metadata = self.train_and_evaluate(model_spec, self.config, model_dir)
        metadata_file = os.path.join(model_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, cls=_NumpyEncoder)

        data_point = {}
        data_point['module_adjacency'] = model_spec.matrix
        data_point['module_operations'] = model_spec.ops
        data_point['trainable_parameters'] = metadata['trainable_params']

        final_evaluation = metadata['evaluation_results'][-1]
        data_point['training_time'] = final_evaluation['training_time']
        data_point['train_accuracy'] = final_evaluation['train_accuracy']
        data_point['validation_accuracy'] = final_evaluation['validation_accuracy']
        data_point['test_accuracy'] = final_evaluation['test_accuracy']

        return data_point

    def hash_iterator(self):
        """Returns iterator over all unique model hashes."""
        return self.fixed_statistics.keys()

    def get_metrics_from_hash(self, module_hash):
        """Returns the metrics for all epochs and all repeats of a hash.

        This method is for dataset analysis and should not be used for benchmarking.
        As such, it does not increment any of the budget counters.

        Args:
          module_hash: MD5 hash, i.e., the values yielded by hash_iterator().

        Returns:
          fixed stats and computed stats of the model spec provided.
        """
        fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
        computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
        return fixed_stat, computed_stat

    def get_metrics_from_spec(self, model_spec):
        """Returns the metrics for all epochs and all repeats of a model.

        This method is for dataset analysis and should not be used for benchmarking.
        As such, it does not increment any of the budget counters.

        Args:
          model_spec: ModelSpec object.

        Returns:
          fixed stats and computed stats of the model spec provided.
        """
        self._check_spec(model_spec)
        module_hash = self._hash_spec(model_spec)
        return self.get_metrics_from_hash(module_hash)

    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        return model_spec.valid_spec

    def _hash_spec(self, model_spec):
        """Returns the MD5 hash for a provided model_spec."""
        return model_spec.hash_spec()


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy objects to JSON-serializable format."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Matrices converted to nested lists
            return obj.tolist()
        elif isinstance(obj, np.generic):
            # Scalars converted to closest Python type
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)
