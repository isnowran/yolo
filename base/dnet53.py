import torch.nn as nn
import torch
from basenet import *

class darknet53(nn.Module):
    def __init__(self, ver, num_classes, init_weight):
        super(darknet53, self).__init__()
        self.ver = ver
        rsize = 4 + 1 + num_classes

        def RESI(x, n):
            return [resi(x) for _ in range(n)]

        x = 32
        ss = [conv(3, x), down(x, x*2), resi(x*2), down(x*2, x*4)]

        x = 128
        ss.extend([resi(x), resi(x)])
        self.start = nn.Sequential(*ss)

        ss = [down(x, x*2)]

        x = 256
        ss.extend(RESI(x, 8))
        self.stage52 = nn.Sequential(*ss)

        ss = [down(x, x*2)]

        x = 512
        ss.extend(RESI(x, 8))
        self.stage26 = nn.Sequential(*ss)

        ss = [down(x, x*2)]

        x = 1024
        ss.extend(RESI(x, 4))
        self.stage13 = nn.Sequential(*ss)

        bsize = 3 if ver == 'v3' else 5
        self.cset13 = cset(x)
        self.end13 = nn.Sequential(up_k3(x//2), nn.Conv2d(x, bsize*rsize, kernel_size=1, padding=0))

        if ver == 'v3':
            # 26
            x = 512
            cx = 768
            self.up26 = upsample(x)
            self.cset26 = cset(cx)
            self.end26 = nn.Sequential(up_k3(cx//2), nn.Conv2d(cx, bsize*rsize, kernel_size=1, padding=0))

            # 52
            x = 384
            cx = 448
            self.up52 = upsample(x)
            self.cset52 = cset(cx)
            self.end52 = nn.Sequential(up_k3(cx//2), nn.Conv2d(cx, bsize*rsize, kernel_size=1, padding=0))

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0, nonlinearity='leaky_relu')

    def forward(self, imgx):
        out_start = self.start(imgx)
        out52 = self.stage52(out_start)
        out26 = self.stage26(out52)
        out13 = self.stage13(out26)
        set13 = self.cset13(out13)
        end13 = self.end13(set13)

        if self.ver != 'v3':
            return end13

        # up26
        up26 = self.up26(set13)
        cat26 = torch.cat((out26, up26), dim=1)
        set26 = self.cset26(cat26)
        end26 = self.end26(set26)

        # up52
        up52 = self.up52(set26)
        cat52 = torch.cat((out52, up52), dim=1)
        set52 = self.cset52(cat52)
        end52 = self.end52(set52)

        return end13, end26, end52


class darknet53_v3(darknet53):
    def __init__(self, num_classes, init_weight=True):
        super(darknet53_v3, self).__init__('v3', num_classes, init_weight)


class darknet53_v2(darknet53):
    def __init__(self, num_classes, init_weight=True):
        super(darknet53_v2, self).__init__('v2', num_classes, init_weight)


if __name__ == "__main__":
    net_v2 = darknet53_v2(20)
    net_v3 = darknet53_v3(20)
    isize = 416
    try:
        from torchsummary import summary
        summary(net_v3, (3, isize, isize))
    except:
        print('except')

    img = torch.rand(2, 3, isize, isize)
    r13, r26, r52 = net_v3(img)
    print(r13.shape, r26.shape, r52.shape)

    r13 = net_v2(img)
    print(r13.shape)
