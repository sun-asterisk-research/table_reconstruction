from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d

from .unet_parts import OutConv, Up


def conv(ni, nf, ks=3, stride=1, act=True, bn=True):
    layers = []
    layers.append(
        nn.Conv2d(
            ni, nf, kernel_size=ks, padding=(ks - 1) // 2, stride=stride, bias=False
        )
    )
    if act:
        layers.append(nn.ReLU(inplace=True))
    if bn:
        layers.append(BatchNorm2d(nf))

    return nn.Sequential(*layers)


def conv_block(ni, nf, stride):
    return nn.Sequential(
        conv(ni, nf // 4, ks=1),
        conv(nf // 4, nf // 4, stride=stride),
        conv(nf // 4, nf, ks=1, bn=True, act=False),
    )


def _resnet_stem(*sizes):
    convs = [
        conv(sizes[i], sizes[i + 1], ks=3, stride=2 if i == 0 else 1)
        for i in range(len(sizes) - 1)
    ]
    return nn.Sequential(*convs)


def noop(x):
    return x


def block(ni, nf, idx, stride=2, nblocks=2):
    return nn.Sequential(
        *[
            ResBlock(ni if i == 0 else nf, nf, stride=stride if i == 0 else 1)
            for i in range(nblocks)
        ]
    )


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super(ResBlock, self).__init__()
        self.convs = conv_block(ni, nf, stride)
        self.idconv = noop if ni == nf else conv(ni, nf, 1, act=None)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))


class ResUnet(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(ResUnet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.stem = _resnet_stem(3, 32, 32, 64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = block(64, 64, 0, stride=1, nblocks=3)
        self.block2 = block(64, 128, 0, stride=2, nblocks=4)
        self.block3 = block(128, 256, 0, stride=2, nblocks=6)
        self.block4 = block(256, 512, 0, stride=2, nblocks=3)

        self.up6 = Up(512 + 256, 256, bilinear)
        self.up5 = Up(256 + 128, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)
        self.up3 = Up(64 + 64, 64, bilinear)
        self.up2 = Up(64 + 64, 32, bilinear)
        self.up1 = Up(32 + 3, 32, bilinear)

        self.outconv = OutConv(32, n_classes)

    def forward(self, x):
        d1 = self.stem(x)  # 64
        d2 = self.pool(d1)

        d3 = self.block1(d2)  # 64
        d4 = self.block2(d3)  # 128
        d5 = self.block3(d4)  # 256
        d6 = self.block4(d5)  # 512

        h = self.up6(d6, d5)  # 512 + 256 -> 256
        h = self.up5(h, d4)  # 256 + 128 -> 128
        h = self.up4(h, d3)  #
        h = self.up3(h, d2)
        h = self.up2(h, d1)
        h = self.up1(h, x)

        h = self.outconv(h)
        return h
