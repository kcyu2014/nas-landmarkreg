import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from monodepth.models.blocks import Interpolate


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, 2 * planes, kernel_size=1)

        self.bn = nn.BatchNorm2d(2 * planes)

        self.atrous_conv = nn.Conv2d(
            2 * planes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self._init_weight()

    def forward(self, x):
        x = self.bn(self.conv1(F.relu(x)))
        x = self.atrous_conv(F.relu(x))

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ASPPLayer(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super(ASPPLayer, self).__init__()

        self.upconv = _make_upconv(2*outfeatures, outfeatures, use_batchnorm=True)      
        self.conv   = _make_conv(infeatures + outfeatures, outfeatures, use_batchnorm=True)
        
        self.rate3 = _ASPPModule(
            outfeatures, outfeatures // 2, kernel_size=3, padding=3, dilation=3
        )

        self.rate6 = _ASPPModule(
            192 + outfeatures + outfeatures // 2,
            outfeatures // 2,
            kernel_size=3,
            dilation=6,
            padding=6,
        )

        self.rate12 = _ASPPModule(
            192 + outfeatures + outfeatures,
            outfeatures // 2,
            kernel_size=3,
            dilation=6,
            padding=6,
        )

        self.rate18 = _ASPPModule(
            192 + outfeatures + outfeatures + outfeatures // 2,
            outfeatures // 2,
            kernel_size=3,
            dilation=6,
            padding=6,
        )

        self.rate24 = _ASPPModule(
            192 + outfeatures + outfeatures + outfeatures,
            outfeatures // 2,
            kernel_size=3,
            dilation=6,
            padding=6,
        )

        # Rest of decoder starts here
        self.output = _make_conv(
            outfeatures + 5 * (outfeatures // 2), outfeatures // 2
        )
        
    def forward(self, x0, x1):
        """Forward pass.

        Args:
            x0 (tensor): low resolution input
            x1 (tensor): lower resolution input (half x0)

        Returns:
            tensor: aspp features
        """

        concat = torch.cat((self.upconv(x1), x0), 1)
        rate1 = self.conv(concat)

        rate3 = self.rate3(rate1)

        concat2 = torch.cat((concat, rate3), 1)
        rate6 = self.rate6(concat2)

        concat3 = torch.cat((concat2, rate6), 1)
        rate12 = self.rate12(concat3)

        concat4 = torch.cat((concat3, rate12), 1)
        rate18 = self.rate18(concat4)

        concat5 = torch.cat((concat4, rate18), 1)
        rate24 = self.rate24(concat5)

        concat_daspp = torch.cat(
            (rate1, rate3, rate6, rate12, rate18, rate24), 1
        )

        return self.output(concat_daspp)



def _make_conv(infeatures, outfeatures, use_batchnorm=False):
    blocks = [nn.Conv2d(infeatures, outfeatures, kernel_size=3, stride=1, padding=1),
              nn.ELU()]

    if use_batchnorm:
        blocks.append(nn.BatchNorm2d(outfeatures))
        
    return nn.Sequential(*blocks)


def _make_upconv(infeatures, outfeatures, use_batchnorm=False):

    blocks = [nn.Conv2d(infeatures, outfeatures, kernel_size=3, stride=1, padding=1),
              Interpolate(2.0, "nearest")]

    if use_batchnorm:
        blocks.append(nn.BatchNorm2d(outfeatures))
        
    return nn.Sequential(*blocks)


class PlanarNet(nn.Module):
    def __init__(self, path=None, features=512):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
        """
        super(PlanarNet, self).__init__()

        use_pretrained = False if path else True
        densenet = models.densenet161(pretrained=use_pretrained).features

        # Backbone
        self.pretrained = nn.Module()
        self.pretrained.h_2 = nn.Sequential(
            densenet.conv0, densenet.norm0, densenet.relu0
        )

        self.pretrained.h_4 = densenet.pool0
        self.pretrained.h_8 = nn.Sequential(densenet.denseblock1, densenet.transition1)
        self.pretrained.h_16 = nn.Sequential(densenet.denseblock2, densenet.transition2)
        self.pretrained.h_32 = nn.Sequential(
            densenet.denseblock3, densenet.transition3, densenet.denseblock4
        )

        # Initial upsampling to H/8
        self.scratch = nn.Module()
        self.scratch.upconv5 = _make_upconv(2208, features, use_batchnorm=True)
        self.scratch.iconv5 = _make_conv(384 + features, features)

        features = features // 2

        # --- ASPP layers ----
        self.scratch.aspp = ASPPLayer(192, features)
        
        features = features // 2
        self.scratch.upconv3 = _make_upconv(features, features, use_batchnorm=True)        
        self.scratch.iconv3 = _make_conv(96 + features, features)

        features = features // 2
        self.scratch.upconv2 = _make_upconv(2 * features, features)
        self.scratch.iconv2 = _make_conv(96 + features, features)

        features = features // 2
        self.scratch.upconv1 = _make_upconv(2 * features, features)
        self.scratch.iconv1 = _make_conv(features, features)

        # Output decoder
        self.scratch.output_conv = nn.Sequential(
            nn.ELU(), nn.Conv2d(features, 1, kernel_size=3, padding=1)
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image2

        Returns:
            tensor: depth
        """

        f_2  = self.pretrained.h_2(x)
        f_4  = self.pretrained.h_4(f_2)
        f_8  = self.pretrained.h_8(f_4)
        f_16 = self.pretrained.h_16(f_8)
        f_32 = self.pretrained.h_32(f_16)

        upconv5 = self.scratch.upconv5(f_32)
        h_16 = self.scratch.iconv5(torch.cat((upconv5, f_16), 1))
        
        aspp_feat = self.scratch.aspp(f_8, h_16)
        
        h_4 = torch.cat((self.scratch.upconv3(aspp_feat), f_4), 1)
        h_4 = self.scratch.iconv3(h_4)

        h_2 = torch.cat((self.scratch.upconv2(h_4), f_2), 1)
        h_2 = self.scratch.iconv2(h_2)

        out = self.scratch.upconv1(h_2)
        out = self.scratch.iconv1(out)
        out = self.scratch.output_conv(out)

        return torch.squeeze(out, dim=1)
