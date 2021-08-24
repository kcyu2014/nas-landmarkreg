"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from monodepth.models.blocks import (
    DJFCoordFeatureFusionBlock,
    DJFFeatureFusionBlock,
    Interpolate,
    ASPP,
    _make_encoder,
)

from monodepth.models.base_model import BaseModel


class PacNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, backbone="resnet50", use_coord=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(PacNet, self).__init__()

        use_pretrained = False if path else True

        self.pretrained, self.scratch = _make_encoder(
            backbone, features, use_pretrained
        )

        self.scratch.guidance = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=1),
        )

        if use_coord:
            self.scratch.refinenet4 = DJFCoordFeatureFusionBlock(features)
            self.scratch.refinenet3 = DJFCoordFeatureFusionBlock(features)
            self.scratch.refinenet2 = DJFCoordFeatureFusionBlock(features)
            self.scratch.refinenet1 = DJFCoordFeatureFusionBlock(features)
        else:
            self.scratch.refinenet4 = DJFFeatureFusionBlock(features)
            self.scratch.refinenet3 = DJFFeatureFusionBlock(features)
            self.scratch.refinenet2 = DJFFeatureFusionBlock(features)
            self.scratch.refinenet1 = DJFFeatureFusionBlock(features)

        self.scratch.output1 = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
        )

        ## Complex decoder
        self.scratch.output2 = nn.Sequential(
            nn.Conv2d(128 + 40, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

        self.aspp = ASPP()

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
        layer_4_rn = self.aspp(self.pretrained.layer4(layer_3))

        guidance = self.scratch.guidance(x)

        guidance_4 = nn.functional.interpolate(
            guidance, size=layer_3.shape[2:], mode="bilinear"
        )
        guidance_3 = nn.functional.interpolate(
            guidance, size=layer_2.shape[2:], mode="bilinear"
        )
        guidance_2 = nn.functional.interpolate(
            guidance, size=layer_1.shape[2:], mode="bilinear"
        )

        guidance_1 = nn.functional.interpolate(
            guidance, scale_factor=0.5, mode="bilinear"
        )

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        # layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(guidance_4, layer_4_rn)
        path_3 = self.scratch.refinenet3(guidance_3, path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(guidance_2, path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(guidance_1, path_2, layer_1_rn)

        out = self.scratch.output1(path_1)
        out = self.scratch.output2(torch.cat((out, guidance), 1))

        return torch.squeeze(out, dim=1)
