import math
import warnings
from config import device
import torch
import torch.nn as nn
import torch.nn.functional as F


class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class pointwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer=1):
        super().__init__()

        self.dwc = nn.Sequential(nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin),
                                 nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
                                 )

    def forward(self, x):
        out = self.dwc(x)
        return out


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True, activ=True):
        super(double_conv, self).__init__()

        ops = []

        # original
        # ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        # FIXME
        ops += [depthwise_separable_conv(in_ch, out_ch)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]

        # original
        # ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        # FIXME
        ops += [depthwise_separable_conv(out_ch, out_ch)]

        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True, ceil=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=ceil),
            double_conv(in_ch, out_ch, normaliz=normaliz)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True, activ=True):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        # self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch,
                                normaliz=normaliz, activ=activ)

    def forward(self, x1, x2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Upsample is deprecated
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2)),
                        diffY // 2, int(math.ceil(diffY / 2))))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 1),
        # )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_100(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 device=torch.device('cuda')):

        super(UNet_100, self).__init__()
        self.device = device

        self.inc = inconv(n_channels, 64)

        self.down0_1 = down(64, 64)
        self.down0_2 = down(64, 128)
        self.down0_3 = down(128, 256, ceil=True)

        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512, normaliz=False)
        self.down7 = down(512, 512, normaliz=False)

        self.up1 = up(1024, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 256)
        self.up4 = up(512, 256)
        self.up5 = up(512, 256)
        self.up6 = up(512, 256, activ=False)

        self.outc = outconv(256, n_classes)
        self.out_nonlin = nn.Sigmoid()
        self.branch_2 = nn.Sequential(nn.Linear(100 * 100, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(p=0.5))
        self.regressor = nn.Sequential(nn.Linear(64, 1),
                                       nn.ReLU())

        self.lin = nn.Linear(1, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        batch_size = x.shape[0]

        x1_ = self.inc(x)             # 800  64

        x1_ = self.down0_1(x1_)       # 400  64
        x1_ = self.down0_2(x1_)       # 200  128
        x1 = self.down0_3(x1_)        # 100  256

        x2 = self.down1(x1)           # 50  256
        x3 = self.down2(x2)           # 25  256
        x4 = self.down3(x3)           # 12  512
        x5 = self.down4(x4)           # 6   512
        x6 = self.down5(x5)           # 3   512
        x7 = self.down6(x6)           # 1   512

        x = self.up1(x7, x6)          # 3   512
        x = self.up2(x, x5)           # 6   512
        x = self.up3(x, x4)           # 12  512
        x = self.up4(x, x3)           # 25  256
        x = self.up5(x, x2)           # 50  256
        x = self.up6(x, x1)           # 100 256

        x = self.outc(x)

        x_flat = x.view(batch_size, -1)
        x_flat = self.branch_2(x_flat)
        x = self.out_nonlin(x)
        obj = x.squeeze(1)                                      # [B, 100, 100]
        # obj = (obj / obj.sum())
        cnt = self.regressor(x_flat).squeeze(-1)                # [B]

        return obj, cnt


if __name__ == '__main__':
    img = torch.randn([1, 3, 800, 800]).to(device)
    model = UNet_100(n_channels=img.size(1), n_classes=1).to(device)
    outs, counts = model(img)
    print(outs.size())
    print(counts.size())