"""
Implement the small version of NASBench dataset for Monodepth estimation 


"""
import os
import functools
import pandas as pd
from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from peewee import fn
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase
from playhouse.shortcuts import model_to_dict


from .analysis import *

db = SqliteExtDatabase(os.path.join('configs/nasbench201-monodepth.db'), autoconnect=True)
def preprocess_retrain_landmarks_to_csv(root_dirs, output_file):

    # default values
    root_dirs = root_dirs or [
        'random-nonzero/nasbench201/mutator_nonzero-epochs300-lr0.0001',
        'random-nonzero/nasbench201-upsample/mutator_nonzero-epochs300-lr0.0001',
        'random-nonzero-sync/nasbench201/mutator_nonzero-sync-epochs300-lr0.0001',
        'random-nonzero-sync/nasbench201-upsample/mutator_nonzero-sync-epochs300-lr0.0001',
        ]
    output_file = output_file or 'configs/monodepth-nasbench201-2020-08-27.csv'

    # let plot something for the meeting :D 
    # plot_data = pd.DataFrame(columns=['arch', 'space', 'train_loss', 'valid_loss', 'params', 'group', 'epochs'])

    pd_data_frames = []
    for group_id, root in enumerate(root_dirs):
        space_name = root.split('/')[1]
        for arch_id in range(30):
            res = collect_experiment_result(f'experiments/landmarks/{root}/{arch_id}')
            num_epoch, train_loss, valid_loss, num_param, arch, all_train, all_valid = res
            pd_data_frames.append(
                pd.DataFrame([[arch, space_name, train_loss, valid_loss, num_param, 
                'sync' if 'sync' in root else 'normal', 
                'nonzero',
                num_epoch, all_train, all_valid],], 
                columns=['arch', 'space', 'train_loss', 'valid_loss', 'params', 'mode', 'sampler', 
                'epochs', 'all_train_loss', 'all_valid_loss'])
            )
    plot_data  = pd.concat(pd_data_frames, ignore_index=True)
    plot_data.to_csv(output_file)


def update_nb201_dataset(csv_path='configs/monodepth-nasbench201-2020-08-27.csv', iteration=0):
    print('Loading dataset from csv_path: ', csv_path)
    data = pd.read_csv(csv_path)
    print(f'Finished with {len(data)} data points...' )
    if len(db.get_tables()) < 3:
        db.create_tables([Nb201TrialConfig, Nb201TrialStats, Nb201IntermediateStats])

    for ind in range(len(data)):
        d = data.iloc[ind]
        data_parsed ={
            'arch': d['arch'],
            'num_epochs': d['epochs'],
            'dataset': 'redweb',
            'num_channels': 256,
            'num_cells': 4,
            'iteration': iteration, 
            'space': d['space'],
            'mode': d['mode'],
            'sampler': d['sampler']
        }
        config = Nb201TrialConfig.create(
            # arch=d['arch'], num_epochs=d['epochs'], dataset='redweb', num_channels=256, num_cells=4, 
            **data_parsed
            )
        # parse the trial stat data
        data_parsed = {
                                'train_acc': None,
                                'valid_acc': None,
                                'test_acc': None,
                                'ori_test_acc': None,
                                'train_loss': d['train_loss'],
                                'valid_loss': d['valid_loss'],
                                'test_loss': None,
                                'ori_test_loss': None,
                                'parameters': d['params'],
                                'flops': None,
                                'latency': None,
                                'training_time': None,
                                'valid_evaluation_time': None,
                                'test_evaluation_time': None,
                                'ori_test_evaluation_time': None,
                            }

        trial_stats = Nb201TrialStats.create(config=config, seed=0, **data_parsed)
        intermediate_stats = []
        for epoch in range(d['epochs']):
            # parse intermediate stat
            data_parsed = {
                    'train_acc': None,
                    'valid_acc': None,
                    'test_acc': None,
                    'ori_test_acc': None,
                    'train_loss': d['all_train_loss'][epoch],
                    'valid_loss': d['all_valid_loss'][epoch],
                    'test_loss': None,
                    'ori_test_loss': None,
            }
            data_parsed.update(current_epoch=epoch+1, trial=trial_stats)
            intermediate_stats.append(data_parsed)
        Nb201IntermediateStats.insert_many(intermediate_stats).execute(db)


def query_nb201_trial_stats(arch, num_epochs, dataset, reduction=None, other_conditions=None):
    """
    Query trial stats of NAS-Bench-201 given conditions.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nasbench201.Nb201TrialConfig`. Only trial stats
        matched will be returned. If none, architecture will be a wildcard.
    num_epochs : int or None
        If int, matching results will be returned. Otherwise a wildcard.
    dataset : str or None
        If specified, can be one of the dataset available in :class:`nni.nas.benchmark.nasbench201.Nb201TrialConfig`.
        Otherwise a wildcard.
    reduction : str or None
        If 'none' or None, all trial stats will be returned directly.
        If 'mean', fields in trial stats will be averaged given the same trial config.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nasbench201.Nb201TrialStats` objects,
        where each of them has been converted into a dict.
    """
    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in Nb201TrialStats._meta.sorted_field_names:
            if field_name not in ['id', 'config', 'seed']:
                fields.append(fn.AVG(getattr(Nb201TrialStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(Nb201TrialStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = Nb201TrialStats.select(*fields, Nb201TrialConfig).join(Nb201TrialConfig)
    conditions = []
    if arch is not None:
        conditions.append(Nb201TrialConfig.arch == arch)
    if num_epochs is not None:
        conditions.append(Nb201TrialConfig.num_epochs == num_epochs)
    if dataset is not None:
        conditions.append(Nb201TrialConfig.dataset == dataset)
    if other_conditions:
        conditions.extend(other_conditions)
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(Nb201TrialStats.config)
    for k in query:
        yield model_to_dict(k)


class Nb201TrialConfig(Model):
    """
    Trial config for NAS-Bench-201.
    Attributes
    ----------
    arch : dict
        A dict with keys ``0_1``, ``0_2``, ``0_3``, ``1_2``, ``1_3``, ``2_3``, each of which
        is an operator chosen from :const:`nni.nas.benchmark.nasbench201.NONE`,
        :const:`nni.nas.benchmark.nasbench201.SKIP_CONNECT`,
        :const:`nni.nas.benchmark.nasbench201.CONV_1X1`,
        :const:`nni.nas.benchmark.nasbench201.CONV_3X3` and :const:`nni.nas.benchmark.nasbench201.AVG_POOL_3X3`.
    num_epochs : int
        Number of epochs planned for this trial. Should be one of 12 and 200.
    num_channels: int
        Number of channels for initial convolution. 16 by default.
    num_cells: int
        Number of cells per stage. 5 by default.
    dataset: str
        Dataset used for training and evaluation. 
        redweb indicate the ReDWeb dataset used to train mono-depth task.
    """

    arch = JSONField(index=True)
    num_epochs = IntegerField(index=True)
    num_channels = IntegerField()
    num_cells = IntegerField()
    iteration = IntegerField()
    mode = CharField(max_length=20, choices=['normal', 'sync'])
    space = CharField(max_length=20, choices=['v1', 'v1+upsample'])
    dataset = CharField(max_length=20, index=True, choices=[
        'redweb',  # 25k+25k+10k
    ])

    class Meta:
        database = db


class Nb201TrialStats(Model):
    """
    Computation statistics for NAS-Bench-201. Each corresponds to one trial.
    Attributes
    ----------
    config : Nb201TrialConfig
        Setup for this trial data.
    seed : int
        Random seed selected, for reproduction.
    train_acc : float
        Final accuracy on training data, ranging from 0 to 100.
    valid_acc : float
        Final accuracy on validation data, ranging from 0 to 100.
    test_acc : float
        Final accuracy on test data, ranging from 0 to 100.
    ori_test_acc : float
        Test accuracy on original validation set (10k for CIFAR and 12k for Imagenet16-120),
        ranging from 0 to 100.
    train_loss : float or None
        Final cross entropy loss on training data. Note that loss could be NaN, in which case
        this attributed will be None.
    valid_loss : float or None
        Final cross entropy loss on validation data.
    test_loss : float or None
        Final cross entropy loss on test data.
    ori_test_loss : float or None
        Final cross entropy loss on original validation set.
    parameters : float
        Number of trainable parameters in million.
    latency : float
        Latency in seconds.
    flops : float
        FLOPs in million.
    training_time : float
        Duration of training in seconds.
    valid_evaluation_time : float
        Time elapsed to evaluate on validation set.
    test_evaluation_time : float
        Time elapsed to evaluate on test set.
    ori_test_evaluation_time : float
        Time elapsed to evaluate on original test set.
    """
    config = ForeignKeyField(Nb201TrialConfig, backref='trial_stats', index=True)
    seed = IntegerField()
    train_acc = FloatField(null=True)
    valid_acc = FloatField(null=True)
    test_acc = FloatField(null=True)
    ori_test_acc = FloatField(null=True)  # test accuracy of the original test set
    train_loss = FloatField()  # possibly nan
    valid_loss = FloatField()
    test_loss = FloatField(null=True)
    ori_test_loss = FloatField(null=True)
    parameters = FloatField()  # parameters in million
    latency = FloatField(null=True)  # latency in milliseconds
    flops = FloatField(null=True)  # flops in million
    training_time = FloatField(null=True)
    valid_evaluation_time = FloatField(null=True)
    test_evaluation_time = FloatField(null=True)
    ori_test_evaluation_time = FloatField(null=True)

    class Meta:
        database = db


class Nb201IntermediateStats(Model):
    """
    Intermediate statistics for NAS-Bench-201.
    Attributes
    ----------
    trial : Nb201TrialStats
        Corresponding trial.
    current_epoch : int
        Elapsed epochs.
    train_acc : float
        Current accuracy on training data, ranging from 0 to 100.
    valid_acc : float
        Current accuracy on validation data, ranging from 0 to 100.
    test_acc : float
        Current accuracy on test data, ranging from 0 to 100.
    ori_test_acc : float
        Test accuracy on original validation set (10k for CIFAR and 12k for Imagenet16-120),
        ranging from 0 to 100.
    train_loss : float or None
        Current cross entropy loss on training data.
    valid_loss : float or None
        Current cross entropy loss on validation data.
    test_loss : float or None
        Current cross entropy loss on test data.
    ori_test_loss : float or None
        Current cross entropy loss on original validation set.
    """

    trial = ForeignKeyField(Nb201TrialStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_acc = FloatField(null=True)
    valid_acc = FloatField(null=True)
    test_acc = FloatField(null=True)
    ori_test_acc = FloatField(null=True)
    train_loss = FloatField()
    valid_loss = FloatField()
    test_loss = FloatField(null=True)
    ori_test_loss = FloatField(null=True)

    class Meta:
        database = db