import torch
import torch.nn as nn

from monodepth.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from monodepth.models.aspp import build_aspp
from monodepth.models.decoder import build_decoder
from monodepth.models.backbone import build_backbone
from monodepth.models.blocks import Interpolate


class DeepLab(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        output_stride=16,
        features=256,
        sync_bn=True,
        freeze_bn=False,
    ):
        super(DeepLab, self).__init__()
        if backbone == "drn":
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.pretrained = build_backbone(backbone, output_stride, BatchNorm)

        self.scratch = nn.Module()
        self.scratch.layer1 = build_aspp(backbone, output_stride, BatchNorm)
        self.scratch.layer2 = build_decoder(features, backbone, BatchNorm)

        # adaptive output module: 2 convolutions and upsampling
        self.scratch.output_conv1 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            #            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.pretrained(input)
        x = self.scratch.layer1(x)
        x = self.scratch.layer2(x, low_level_feat)
        x = self.scratch.output_conv1(x)
        x = self.scratch.output_conv2(x)

        return x.squeeze(1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone="drn", output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output.size())
