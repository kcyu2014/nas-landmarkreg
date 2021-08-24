import torch
import torch.nn as nn
from torch.optim import SGD

import numpy as np


def copy_pytorch_model_parameters(model_source, model_target):
    """
    Implement with load state dict.
    to save and copy all the weights, without any issue.

    :param model_source:
    :param model_target:
    :return:
    """
    model_source = model_source if isinstance(model_source, nn.Module) else None
    model_target.load_state_dict(model_source.state_dict())
    #
    # for pa, pb in zip(model_source.parameters(), model_target.parameters()):
    #     pb.data = pa.data.clone()
    return model_target


def compare_two_models(model_a, model_b, in_data, out_data, num_repeat=3, test_backward=True):
    # compare the output of the model
    # Check of results match

    # random n times

    compare_forward = 0
    # compare_backward = 0

    opt_a = SGD(model_a.parameters(), lr=0.01)
    opt_b = SGD(model_b.parameters(), lr=0.01)

    for epoch in range(num_repeat):
        in_data = torch.rand(*in_data.shape)

        result_a = model_a(in_data)
        result_b = model_b(in_data)
        compare_forward += np.mean(np.abs(
            result_a.clone().cpu().detach().numpy() -
            result_b.clone().cpu().detach().numpy()))
        if test_backward:
            opt_a.zero_grad()
            loss_a = nn.CrossEntropyLoss()(result_a, out_data)
            loss_a.backward()
            opt_a.step()

            opt_b.zero_grad()
            loss_b = nn.CrossEntropyLoss()(result_b, out_data)
            loss_b.backward()
            opt_b.step()

        assert compare_forward > 1e-6, "failing at iteration {}".format(epoch)

