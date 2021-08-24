"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from monodepth.models.blocks import (
    ASPP,
    FeatureFusionBlock,
    Interpolate,
    _make_encoder,
    ProgressiveUpsample,
    DJFFeatureFusionBlock,
)

from monodepth.models.base_model import BaseModel


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, backbone="resnet50", non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights to MidasNet: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path else True

        self.pretrained, self.scratch = _make_encoder(
            backbone, features, use_pretrained
        )
        
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

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

        # guidance = self.scratch.guidance(x)
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)


class DJFMidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, backbone="resnet50", non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(DJFMidasNet, self).__init__()

        use_pretrained = False if path else True

        self.pretrained, self.scratch = _make_encoder(
            backbone, features, use_pretrained
        )

        self.scratch.refinenet4 = DJFFeatureFusionBlock(features)
        self.scratch.refinenet3 = DJFFeatureFusionBlock(features)
        self.scratch.refinenet2 = DJFFeatureFusionBlock(features)
        self.scratch.refinenet1 = DJFFeatureFusionBlock(features)

        self.scratch.guidance = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        )

        self.scratch.output1 = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
        )

        self.scratch.output2 = nn.Sequential(
            nn.Conv2d(128 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

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

        # guidance = self.scratch.guidance(x)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

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
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(guidance_4, layer_4_rn)
        path_3 = self.scratch.refinenet3(guidance_3, path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(guidance_2, path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(guidance_1, path_2, layer_1_rn)

        out = self.scratch.output1(path_1)
        out = self.scratch.output2(torch.cat((out, guidance), 1))

        return torch.squeeze(out, dim=1)


class MidasNet_StackedHourGlass(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, backbone="resnet50", non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(MidasNet_StackedHourGlass, self).__init__()

        use_pretrained = False if path else True

        self.pretrained, self.scratch = _make_encoder(
            backbone, features, use_pretrained
        )

        self.scratch.refinenet4 = ProgressiveUpsample(features, features // 2, 32)
        self.scratch.refinenet3 = ProgressiveUpsample(
            features + features // 4, features // 2, 16
        )
        self.scratch.refinenet2 = ProgressiveUpsample(
            features + features // 4, features // 2, 8
        )
        self.scratch.refinenet1 = ProgressiveUpsample(
            features + features // 4, features // 2, 4
        )

        # self.scratch.refinenet3 = FeatureFusionBlock(features)
        # self.scratch.refinenet2 = FeatureFusionBlock(features)
        # self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(32 + 4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

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

        # guidance = self.scratch.guidance(x)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4, ups_4 = self.scratch.refinenet4(layer_4_rn)
        path_3, ups_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2, ups_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1, ups_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # print(path_1.shape)
        # exit()

        ups_0 = self.scratch.output_conv1(path_1)
        out = self.scratch.output_conv2(
            torch.cat((ups_0, ups_1, ups_2, ups_3, ups_4), dim=1)
        )

        out = (
            torch.squeeze(out, dim=1),
            torch.squeeze(ups_1, dim=1),
            torch.squeeze(ups_2, dim=1),
            torch.squeeze(ups_3, dim=1),
            torch.squeeze(ups_4, dim=1),
        )

        return out


class MidasNet_ASPP(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        super(MidasNet_ASPP, self).__init__()

        use_pretrained = False if path else True

        self.pretrained, self.scratch = _make_encoder(
            "resnext101_wsl_aspp", features, use_pretrained
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
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

        # guidance = self.scratch.guidance(x)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        # layer_4 = self.pretrained.layer4(layer_3)
        # print(x.shape)
        # print(layer_4.shape)
        # exit()

        layer_4 = self.aspp(self.pretrained.layer4(layer_3))

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)
