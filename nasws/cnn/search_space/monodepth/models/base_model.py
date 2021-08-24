import torch
import torch.nn as nn


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)
        # if 'model' in parameters.keys():
            # parameters = parameters['model']

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

    # def train(self, mode):
    #     super().train(mode)

    #     if False:
    #         for m in self.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eval()
    #                 if True:
    #                     m.weight.requires_grad = False
    #                     m.bias.requires_grad = False

    #         print("Found batchnorm", m.weight.requires_grad, m.training)

    # exit()
