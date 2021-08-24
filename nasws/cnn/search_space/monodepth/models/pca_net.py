"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from torchvision import models

# from monodepth.models.pac import PacConv2d


class PCANet(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
        """
        super().__init__()

        if path:
            use_pretrained = False
        else:
            use_pretrained = True

        resnet = models.resnet50(pretrained=use_pretrained)

        self.pretrained = nn.Module()
        self.scratch = nn.Module()
        self.pretrained.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )

        self.pretrained.layer2 = resnet.layer2
        self.pretrained.layer3 = resnet.layer3
        self.pretrained.layer4 = resnet.layer4

        with open("data/redweb_pca_basis.pkl", "rb") as pkl:
            pca_data = pickle.load(pkl)
            # self.pca_mean = torch.from_numpy(pca_data["mean"])
            # self.pca_basis = torch.from_numpy(pca_data["components"])

            self.register_buffer(
                "pca_mean", torch.from_numpy(pca_data["mean"]).float().view(1, 1, -1)
            )
            self.register_buffer(
                "pca_basis",
                torch.from_numpy(pca_data["components"]).float().unsqueeze(0),
            )

            self.register_buffer(
                "pca_variance",
                torch.from_numpy(pca_data["explained_variance"]).float().unsqueeze(0),
            )

            #            print(self.pca_basis.shape, self.pca_variance.shape)
            self.pca_basis = (
                torch.sqrt(self.pca_variance.unsqueeze(-1)) * self.pca_basis
            )

        self.scratch.fc1 = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(True)  # nn.BatchNorm1d(1024)
        )
        self.scratch.fc2 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(True)  # nn.BatchNorm1d(512)
        )
        self.scratch.fc3 = nn.Linear(512, 324)

        if path:
            self.load(path)

        if True:
            for m in self.pretrained.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if True:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        latent = F.adaptive_avg_pool2d(layer_4, (1, 1)).view(-1, 2048)

        code = self.scratch.fc1(latent)
        code = self.scratch.fc2(code)
        code = self.scratch.fc3(code).unsqueeze(1)

        basis = self.pca_basis.expand(code.shape[0], *self.pca_basis.shape[1:])
        recon = torch.bmm(code, basis) + self.pca_mean
        out = recon.view(-1, *x.shape[2:])

        return out, code

        # return out

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        self.load_state_dict(parameters)
