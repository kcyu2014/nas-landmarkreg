import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from efficientnet_pytorch import EfficientNet

# from .pac import PacConvTranspose2d, PacConv2d


def _make_encoder(backbone, features, use_pretrained):
    if backbone == "resnet50":
        pretrained = _make_pretrained_resnet50(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features)
    elif backbone == "resnet101":
        pretrained = _make_pretrained_resnet101(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features)
    elif backbone == "densenet161":
        pretrained = _make_pretrained_densenet161(use_pretrained)
        scratch = _make_scratch([384, 768, 2112, 2208], features)
    elif backbone == "resnext101":
        pretrained = _make_pretrained_resnext101_32x8d(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features)
    elif backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features)
    elif backbone == "resnext101_wsl_aspp":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 256], features)
    elif backbone == "efficientnet":
        pretrained = _make_pretrained_efficientnet(use_pretrained)
        scratch = _make_scratch([48, 80, 160, 640], features)
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained, scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnet101(use_pretrained):
    resnet = models.resnet101(pretrained=use_pretrained)
    return _make_resnet_backbone(resnet)


def _make_pretrained_resnet50(use_pretrained):
    resnet = models.resnet50(pretrained=use_pretrained)
    return _make_resnet_backbone(resnet)


def _make_pretrained_resnext101_32x8d(use_pretrained):
    resnet = models.resnext101_32x8d(pretrained=use_pretrained)
    return _make_resnet_backbone(resnet)

def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


def _make_pretrained_efficientnet(use_pretrained):
    effnet = EfficientNet.from_pretrained("efficientnet-b7")
    print(effnet)
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet._conv_stem, effnet._bn0, *effnet._blocks[0:5]
    )
    pretrained.layer2 = nn.Sequential(*effnet._blocks[5:12])
    pretrained.layer3 = nn.Sequential(*effnet._blocks[12:19])
    pretrained.layer4 = nn.Sequential(*effnet._blocks[19:])

    return pretrained


def _make_pretrained_densenet161(use_pretrained):
    densenet = models.densenet161(pretrained=use_pretrained)
    densenet = densenet.features

    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        densenet.conv0,
        densenet.norm0,
        densenet.relu0,
        densenet.pool0,
        densenet.denseblock1,
    )

    pretrained.layer2 = nn.Sequential(densenet.transition1, densenet.denseblock2)
    pretrained.layer3 = nn.Sequential(densenet.transition2, densenet.denseblock3)
    pretrained.layer4 = nn.Sequential(densenet.transition3, densenet.denseblock4)

    return pretrained


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        # self.bn1 = nn.BatchNorm2d(features)  # nn.LocalResponseNorm(2)
        # self.gn1 = nn.GroupNorm(16, features)

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        # self.gn2 = nn.GroupNorm(16, features)
        # self.bn2 = nn.BatchNorm2d(features)  # nn.LocalResponseNorm(2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        # out = self.bn1(self.relu(x))
        out = self.relu(x)
        out = self.conv1(out)
        # out = self.gn1(out)

        # out = self.bn2(self.relu(out))
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.gn2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

        # self.resConfUnit = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)
        #     output += self.resConfUnit(xs[1])

        # output = self.resConfUnit(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


import numpy as np


class ProgressiveUpsample(nn.Module):
    def __init__(self, in_features, features, factor):
        super(ProgressiveUpsample, self).__init__()

        self.factor = factor

        ups_factors = [2 ** (i + 1) for i in range(0, int(np.log2(factor)))]

        up_layers = []
        for fact in ups_factors:
            up_layers.append(
                nn.Sequential(
                    Interpolate(2, "bilinear"),
                    nn.Conv2d(
                        features // (fact // 2),
                        features // fact,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(True),
                )
            )

        self.upsample = nn.Sequential(*up_layers)

        down_layers = []
        for fact in ups_factors[:0:-1]:
            down_layers.append(
                nn.Sequential(
                    Interpolate(0.5, "bilinear"),
                    nn.Conv2d(
                        features // fact,
                        features // (fact // 2),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(True),
                )
            )

        self.downsample = nn.Sequential(*down_layers)

        self.dec = nn.Sequential(
            nn.Conv2d(
                features // ups_factors[-1],
                features // ups_factors[-2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(features // ups_factors[-2], 1, kernel_size=1, padding=0),
            # nn.ReLU(True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(in_features, features, kernel_size=1), nn.ReLU(True)
        )

    def forward(self, *x):
        if len(x) == 2:
            x = self.project(torch.cat(x, dim=1))
        else:
            x = self.project(x[0])

        ups = self.upsample(x)

        return self.downsample(ups), self.dec(ups)


class PACFeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(PACFeatureFusionBlock, self).__init__()

        self.resConfUnit = ResidualConvUnit(features)

        self.upconv = PacConvTranspose2d(
            features,
            features,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
            normalize_kernel=True,
        )

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        guidance = xs[0]
        output = xs[1]

        if len(xs) == 3:
            output += self.resConfUnit(xs[2])

        output = self.resConfUnit(output)
        output = self.upconv(output, guidance)

        return output


class DJFFeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(DJFFeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.inter = nn.Sequential(
            nn.ReLU(True), nn.Conv2d(features + 32, features, kernel_size=1)
        )
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        guidance = xs[0]
        output = xs[1]

        if len(xs) == 3:
            output += self.resConfUnit1(xs[2])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        res = self.resConfUnit2(self.inter(torch.cat((output, guidance), 1)))

        # output = self.upconv(output, guidance)

        return res


class DJFCoordFeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(DJFCoordFeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.inter = nn.Sequential(
            nn.ReLU(True), nn.Conv2d(features + 40 + 1, features, kernel_size=1)
        )
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        guidance = xs[0]
        output = xs[1]

        if len(xs) == 3:
            output += self.resConfUnit1(xs[2])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        yy = torch.linspace(0.0, 1.0, output.shape[2], device=output.device).view(
            1, 1, -1, 1
        )
        yy = yy.expand(output.shape[0], 1, output.shape[2], output.shape[3])

        res = self.resConfUnit2(self.inter(torch.cat((output, guidance, yy), 1)))

        # output = self.upconv(output, guidance)

        return res


class LocalPlanarGuidance(nn.Module):
    def __init__(self, features):
        super(LocalPlanarGuidance, self).__init__()

        self.reduction = nn.Conv2d(features, 4, kernel_size=1)

    def forward(self, x, out_shape):
        u, v = torch.meshgrid(
            [
                torch.linspace(-1, 1, steps=out_shape[0]),
                torch.linspace(-1, 1, steps=out_shape[1]),
            ]
        )
        u = u.to(x.device)
        v = v.to(x.device)

        norm = (u ** 2 + v ** 2 + 1.0).sqrt()
        plane = self.reduction(x)

        plane[:, :3] = F.normalize(torch.tanh(plane[:, :3]), p=2, dim=1)
        plane[:, -1] = torch.sigmoid(plane[:, -1])

        plane = nn.functional.interpolate(plane, size=out_shape, mode="nearest")

        norms = plane[:, :3]
        n4 = plane[:, -1]

        denom = norms[:, 0] * u + norms[:, 1] * v + norms[:, 2]  # .unsqueeze(1)
        output = n4 * norm / denom

        return output.unsqueeze(1)


class LPGFeatureFusionBlock(nn.Module):
    """Feature fusion block with local planar guidance
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(LPGFeatureFusionBlock, self).__init__()

        self.ups_conv = nn.Conv2d(
            2 * features + 1, features, kernel_size=3, stride=1, padding=1
        )
        self.lpg = LocalPlanarGuidance(features)

    def forward(self, x0, x1, out_shape):
        out_upsample = nn.functional.interpolate(
            x0, scale_factor=2, mode="bilinear", align_corners=True
        )

        out_lpg = self.lpg(x0, out_shape)

        out_lpg_small = nn.functional.interpolate(
            out_lpg, size=out_upsample.shape[2:], mode="nearest"
        )

        output = torch.cat((x1, out_upsample, out_lpg_small), 1)
        output = self.ups_conv(output)

        # output = self.project(output)

        return output, out_lpg


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, output_stride=16):  # TODO See what the actual output stride is
        super(ASPP, self).__init__()
        inplanes = 2048

        if output_stride == 32:
            dilations = [1, 3, 6, 9]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[1], dilation=dilations[1]
        )
        self.aspp3 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[2], dilation=dilations[2]
        )
        self.aspp4 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[3], dilation=dilations[3]
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.relu = nn.ReLU(False)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        # x = self.bn1(x)
        #        x = self.relu(x)

        return self.relu(x)  # self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":

    nfeatures = 256
    upconv = PACFeatureFusionBlock(nfeatures)

    prior = torch.randn(1, nfeatures, 12, 12)
    current = torch.randn(1, nfeatures, 12, 12)

    guide = torch.randn(1, 16, 24, 24)

    output = upconv(guide, prior, current)

    print("upsampled=", output.shape)

    # output, out_lpg = lpg(test1, test2, (test1.shape[2] * 8, test1.shape[3] * 8))

    # print(test1.shape)
    # print(test2.shape)
    # print(output.shape)
    # print(out_lpg.shape)
