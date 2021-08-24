import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from torchvision import models, transforms
from monodepth.models.midas_net import FeatureFusionBlock

sys.path.append("external/cnnimageretrieval-pytorch")
sys.path.append("external/SFNet")

from model import SFNet

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.general import get_data_root


class RetrievalNet(nn.Module):
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

        self.scratch.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer4_rn = nn.Conv2d(
            2048, features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv1 = nn.Conv2d(
            features, 128, kernel_size=3, stride=1, padding=1
        )

        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(129, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        # )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(128 + 4, 1, kernel_size=3, stride=1, padding=1)
        )

        if path:
            self.load(path)

        # Freeze batch norm layers
        for m in self.pretrained.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, image, candidates):

        """Forward pass.

        Argsn:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(image)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # do our standard thing
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # upsample
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear")
        fuse = torch.cat((out, candidates), 1)
        out = self.scratch.output_conv2(fuse)

        return out.squeeze(1)

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
