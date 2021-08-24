"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from torchvision import models
from monodepth.models.midas_net import FeatureFusionBlock, ResidualConvUnit

# from monodepth.models.pac import PacConv2d


class NMFNet(nn.Module):
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

        with open("data/redweb_nmf_basis.pkl", "rb") as pkl:
            data = pickle.load(pkl)
            self.register_buffer(
                "nmf_basis", torch.from_numpy(data["components"]).float().unsqueeze(0)
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

        Argsn:
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
        code = F.relu(self.scratch.fc3(code).unsqueeze(1))

        basis = self.nmf_basis.expand(code.shape[0], *self.nmf_basis.shape[1:])
        recon = torch.bmm(code, basis)
        out = recon.view(-1, *x.shape[2:]) * 10

        return out, code.squeeze(1)

        # return out

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        self.load_state_dict(parameters)


class RefineNMFNet(nn.Module):
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

        with open("data/redweb_nmf_basis.pkl", "rb") as pkl:
            data = pickle.load(pkl)
            self.register_buffer(
                "nmf_basis", torch.from_numpy(data["components"]).float().unsqueeze(0)
            )

            # TODO: Also write this out when constructing the basis!
            self.basis_img_size = (192, 320)

        assert (
            self.basis_img_size[0] * self.basis_img_size[1] == self.nmf_basis.shape[-1]
        )

        self.scratch.code_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 324),
            nn.ReLU(True),
        )

        self.scratch.layer1_rn = nn.Conv2d(
            256, features - 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer2_rn = nn.Conv2d(
            512, features - 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer3_rn = nn.Conv2d(
            1024, features - 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer4_rn = nn.Conv2d(
            2048, features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv1 = nn.Conv2d(
            features + 1, 128, kernel_size=3, stride=1, padding=1
        )

        self.scratch.output_conv2 = nn.Conv2d(
            128, 1, kernel_size=3, stride=1, padding=1
        )

        if path:
            self.load(path)

        # Freeze batch norm layers
        for m in self.pretrained.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def reconstruct(self, code):
        basis = self.nmf_basis.expand(code.shape[0], *self.nmf_basis.shape[1:])
        recon = torch.bmm(code, basis)

        return recon.view(-1, 1, *self.basis_img_size) * 10

    def forward(self, x):
        """Forward pass.

        Argsn:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        print(layer_4.shape)
        exit()
        # get basis reconstruction
        latent = F.adaptive_avg_pool2d(layer_4, (1, 1)).view(-1, 2048)
        code = self.scratch.code_layer(latent).unsqueeze(1)

        recon_1 = self.reconstruct(code)

        # Adapt to correct outputsize
        recon_1 = nn.functional.interpolate(recon_1, size=x.shape[2:], mode="bilinear")
        recon_2 = nn.functional.interpolate(recon_1, scale_factor=0.5, mode="bilinear")
        recon_4 = nn.functional.interpolate(recon_2, scale_factor=0.5, mode="bilinear")
        recon_8 = nn.functional.interpolate(recon_4, scale_factor=0.5, mode="bilinear")
        recon_16 = nn.functional.interpolate(recon_8, scale_factor=0.5, mode="bilinear")

        # do our standard thing
        layer_1_rn = torch.cat((self.scratch.layer1_rn(layer_1), recon_4), 1)
        layer_2_rn = torch.cat((self.scratch.layer2_rn(layer_2), recon_8), 1)
        layer_3_rn = torch.cat((self.scratch.layer3_rn(layer_3), recon_16), 1)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # upsample
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        path_1 = torch.cat((path_1, recon_2), 1)

        out = self.scratch.output_conv1(path_1)
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.scratch.output_conv2(out)

        return out.squeeze(1), code.squeeze(1), recon_1.squeeze(1)

        # return out

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


class RefineMultiNMFNet(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, mode="nmf"):
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

        self.whitened = False

        if mode == "pca":
            basis_file = "data/redweb_multipca_basis.pkl"
            self.whitened = True
        elif mode == "nmf":
            basis_file = "data/redweb_multinmf_basis.pkl"
        else:
            assert False

        with open(basis_file, "rb") as pkl:
            data = pickle.load(pkl)

            self.register_buffer(
                "basis", torch.from_numpy(data["all_bases"]).float().unsqueeze(0)
            )

            if mode == "pca":
                self.register_buffer(
                    "means", torch.from_numpy(data["all_means"]).float().unsqueeze(0)
                )
                self.register_buffer(
                    "vars", torch.from_numpy(data["all_vars"]).float().unsqueeze(0)
                )
                self.basis *= self.vars.unsqueeze(-1).sqrt()

                self.vars = self.vars[:, :, :64]
                self.basis = self.basis[:, :, :64]

                # print(self.means.shape)
                # print(self.vars.shape)
                # print(self.basis.shape)
                # exit()

            # TODO: Also write this out when constructing the basis!
            self.basis_img_size = (192, 320)

        assert 32 * 32 == self.basis.shape[-1]

        self.scratch.code_layer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(512, self.basis.shape[2], kernel_size=1, bias=True),
            nn.ReLU(True) if mode == "nmf" else nn.Identity(),
        )

        self.scratch.layer1_rn = nn.Conv2d(
            256, features - 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer2_rn = nn.Conv2d(
            512, features - 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer3_rn = nn.Conv2d(
            1024, features - 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer4_rn = nn.Conv2d(
            2048, features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv1 = nn.Conv2d(
            features + 1, 128, kernel_size=3, stride=1, padding=1
        )

        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(129, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        # )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        )

        if path:
            self.load(path)

        # To reassemble the NMF estimates, assume images of size 192x320
        self.fold = nn.Fold((192, 320), 32, stride=32)

        # Freeze batch norm layers
        for m in self.pretrained.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def reconstruct(self, code):
        code = code
        # Expand the basis to batch size
        basis = (self.basis).expand(code.shape[0], *self.basis.shape[1:])

        # Reshape to cW*cH*BS
        basis = basis.contiguous().view(-1, *basis.shape[2:])

        # Reshape code
        code = code.permute(0, 2, 3, 1).contiguous().view(-1, 1, self.basis.shape[2])

        recon = torch.bmm(code, basis)

        # recon get the batch dimesnion back
        #        print(self.nmf_basis.shape)
        recon = recon.view(-1, self.basis.shape[1], self.basis.shape[-1]).permute(
            0, 2, 1
        )

        if self.whitened:
            recon += self.means.permute(0, 2, 1)

        recon = self.fold(recon)  # ? We shouldn't need this

        return recon

    def forward(self, x):
        """Forward pass.

        Argsn:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        # get basis reconstruction
        code = self.scratch.code_layer(layer_4)

        recon_1 = self.reconstruct(code)

        # Adapt to correct outputsize
        recon_1 = nn.functional.interpolate(recon_1, size=x.shape[2:], mode="bilinear")
        recon_2 = nn.functional.interpolate(recon_1, scale_factor=0.5, mode="bilinear")
        recon_4 = nn.functional.interpolate(recon_2, scale_factor=0.5, mode="bilinear")
        recon_8 = nn.functional.interpolate(recon_4, scale_factor=0.5, mode="bilinear")
        recon_16 = nn.functional.interpolate(recon_8, scale_factor=0.5, mode="bilinear")

        # do our standard thing
        layer_1_rn = torch.cat((self.scratch.layer1_rn(layer_1), recon_4), 1)
        layer_2_rn = torch.cat((self.scratch.layer2_rn(layer_2), recon_8), 1)
        layer_3_rn = torch.cat((self.scratch.layer3_rn(layer_3), recon_16), 1)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # upsample
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        path_1 = torch.cat((path_1, recon_2), 1)

        out = self.scratch.output_conv1(path_1)
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.scratch.output_conv2(out)

        # out = torch.cat((out, recon_1), 1)

        return out.squeeze(1), code.squeeze(1), recon_1.squeeze(1)

        # return out

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
