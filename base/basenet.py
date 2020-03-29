import torch.nn as nn
import torch

def conv(i_channel, o_channel, kernel_size=3, stride=1, pad=1):
    ret = nn.Sequential(nn.Conv2d(i_channel, o_channel, kernel_size, stride, pad, bias=True), \
            nn.BatchNorm2d(o_channel), nn.LeakyReLU(0.1, inplace=False))

    return ret


def down(i_channel, o_channel):
    return conv(i_channel, o_channel, kernel_size=3, stride=2, pad=1)


def down_k1(channel):
    return conv(channel, channel//2, kernel_size=1, stride=1, pad=0)


def up_k3(channel):
    return conv(channel, channel*2, kernel_size=3, stride=1, pad=1)


def cset(x):
    return nn.Sequential(down_k1(x), up_k3(x//2), down_k1(x), up_k3(x//2), down_k1(x))


class upsample(nn.Module):
    def __init__(self, channel):
        super(upsample, self).__init__()
        self.seq = down_k1(channel)

    def forward(self, x):
        return nn.functional.interpolate(self.seq(x), scale_factor=2, mode='bilinear', align_corners=True)


class resi(nn.Module):
    def __init__(self, channel):
        super(resi, self).__init__()
        self.seq = nn.Sequential(down_k1(channel), up_k3(channel//2))

    def forward(self, x):
        return x + self.seq(x)
