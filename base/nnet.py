import torch.nn as nn
import torch
import torchvision as tv
import torch.utils.model_zoo as model_zoo
from snet import s_resnet
import math

class s_yolo(nn.Module):
    def __init__(self):
        super(s_yolo, self).__init__()
        conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=3)
        conv2 = nn.Conv2d(10, 30, kernel_size=5, stride=3)
        self.seq = nn.Sequential(conv1, nn.MaxPool2d(2), conv2, nn.MaxPool2d(2))
        self.isize = 480

    def forward(self, x):
        x = self.seq(x).permute(0, 2, 3, 1)
        return torch.sigmoid(x)

class s_alexnet(tv.models.AlexNet):
    def __init__(self, pretrained, num_classes=1000, linear=True):
        model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

        super(s_alexnet, self).__init__(num_classes)
        self.uselinear = linear
        self.isize = 227 * 2

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

        self.head = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, bias=True, padding=1), nn.MaxPool2d(2))

        if self.uselinear:
            t = 2048
            line1 = nn.Linear(9216, t)
            line2 = nn.Linear(t, 7 * 7 * 30)
            self.seq = nn.Sequential(line1, nn.ReLU(True), nn.Dropout(), line2, nn.Sigmoid())
        else:
            conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
            conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
            conv3 = nn.Conv2d(64, 30, kernel_size=1, stride=1)
            self.seq = nn.Sequential(conv1, nn.ReLU(True), conv2, nn.ReLU(True), conv3, nn.Sigmoid())

    def forward(self, x):
        x = self.head(x)
        x = self.features(x)
        if self.uselinear:
            x = x.view(x.size(0), -1)

        x = self.seq(x)
        if self.uselinear:
            x = x.view(-1, 7, 7, 30)
        else:
            x = x.permute(0, 2, 3, 1)

        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class s_vgg(tv.models.VGG):
    def __init__(self, pretrained, num_classes=1000, linear=True, **kwargs):
        model_urls = {
            'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
            'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
            'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
            'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
            'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        }

        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        if pretrained:
            kwargs['init_weights'] = False

        super(s_vgg, self).__init__(make_layers(cfg['E'], batch_norm=True), **kwargs)

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))

        self.isize = 224 * 2
        self.uselinear = linear
        self.head = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, bias=True, padding=1), nn.ReLU(True), nn.MaxPool2d(2))

        if self.uselinear:
            t = 4096
            line1 = nn.Linear(512 * 7 * 7, t)
            line2 = nn.Linear(t, 7 * 7 * 30)
            self.seq = nn.Sequential(line1, nn.ReLU(True), nn.Dropout(), line2, nn.Sigmoid())
        else:
            conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
            conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
            conv3 = nn.Conv2d(64, 30, kernel_size=1, stride=1)
            self.seq = nn.Sequential(conv1, nn.ReLU(True), conv2, nn.ReLU(True), conv3, nn.Sigmoid())

    def forward(self, x):
        x = self.head(x)
        x = self.features(x)
        if self.uselinear:
            x = x.view(x.size(0), -1)

        x = self.seq(x)
        if self.uselinear:
            x = x.view(-1, 7, 7, 30)
        else:
            x = x.permute(0, 2, 3, 1)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def main():
    import sys
    net = loadnet(sys.argv[1], False)
    if len(sys.argv) > 2:
        checkpoint = sys.argv[2]
        print 'load checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    size = net.isize
    img = torch.rand(2, 3, size, size)
    output = net(img)
    print img.shape, '->', output.shape

if __name__ == '__main__':
    main()
