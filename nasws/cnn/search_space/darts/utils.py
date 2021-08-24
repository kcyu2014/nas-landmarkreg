from nasws.cnn.policy.darts_policy.utils import *
import torch.nn as nn


def change_model_spec(model, spec):
    # this is handling parallel model
    """
    Change model spec, depends on Parallel model or not.
    :param model:
    :param spec:
    :return:
    """
    if isinstance(model, nn.DataParallel):
        model.module.change_genotype(spec.to_darts_genotype())
    else:
        model.change_genotype(spec.to_darts_genotype())

    return model