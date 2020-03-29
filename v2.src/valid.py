import sys
sys.path.append("../")

import os
import torch
from tqdm import trange
from thirdlib import voc_eval
import numpy as np
import fire
import conf

import loader
from base.snet import s_resnet
from base.dnet53 import darknet53_v2 as darknet53
from base import libnet
from base import libiou
from base import util

def writerec(outdir, cmap, cname):
    for k, vs in cmap.iteritems():
        cname.add(k)

        with open("%s/%s" % (outdir, k), 'a+') as fd:
            for v in vs:
                fd.write('%s %.3f %d %d %d %d\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))

def ss(x):
    return x.detach().cpu().numpy().round(4)

def valid(outdir, net, T_p, T_iou, validset, device, drawimg, debug=False):
    cmap = {}
    cset = set()

    for i in trange(len(validset)):
        imgx_ori, imgx, cents, objs, imgid = validset[i]
        imgx_t = imgx.unsqueeze(0).to(device)
        r = net(imgx_t)
        r = libnet.refine(r, conf.class_num+5).squeeze()

        # ap
        h, w = imgx_ori.shape[1:]

        grid = conf.v_resize / 32
        p, rec, maxv, maxind = util.create_box(r, (w, h), grid, conf.anchors.to(device))

        if debug:
            box_all = []
            print 'proc', imgid

            for _y, it0 in enumerate(p):
                for _x, its in enumerate(it0):
                    key = _y, _x

                    for an_indx, _ in enumerate(its):
                        box_all.append((p[_y, _x, an_indx], rec[_y, _x, an_indx], key, maxind[_y, _x, an_indx], maxv[_y, _x, an_indx]))

            points, cellID_all = cents
            has_objs = []
            for _, cell in enumerate(cellID_all):
                key = tuple(cell.tolist())
                has_objs.append(key)

            for _p, _box, _cellID, _cind, _cval in sorted(box_all, key=lambda x: x[0] * x[-1])[-30:]:
                print 'p: %.2f x %.2f = %.2f' % (_p.detach(), _cval.detach(), _p.detach() * _cval.detach()), 'rec', _box.detach().cpu(), 'class', _cind.detach().cpu(), 'cell', _cellID[::-1]

            for ee, obj in enumerate(objs):
                box = libiou.cent2rec((w, h), conf.v_resize/32, cellID_all[ee], points[ee])
                print '-- obj', conf.init_classes.idmap[int(objs[ee])], box, 'cell', cellID_all[ee].tolist()

        box_pure = util.nms(p, rec, maxv, maxind, T_p, T_iou)

        if drawimg:
            util.showimg('%s_%s' % (i, imgid), imgx_ori, box_pure)

        for p, bbox, cval, cind in box_pure:
            cname = conf.init_classes.idmap[int(cind)]
            if cname not in cmap:
                cmap[cname] = []

            cmap[cname].append([imgid, p, bbox[0], bbox[1], bbox[2], bbox[3]])

        if i % 20 == 0:
            writerec(outdir, cmap, cset)
            cmap = {}

    writerec(outdir, cmap, cset)
    return cset

def main(T_p=0.35, T_iou=0.5, checkpoint=None, validroot='vocdata/voc2007_test', cuda='cpu', models=None, maxsize=0, drawimg=False, debug=False):
    outdir = 'output'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    if not models:
        models = util.extact_model(checkpoint)

    if models == 'darknet53':
        net = darknet53(conf.class_num)
    else:
        net = s_resnet(models, False, conf.class_num, 'v2')

    net.eval()

    net = net.to(device)

    if checkpoint:
        print 'load checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    validset = loader.imgloader([validroot], conf.v_resize, grid=conf.v_resize/32, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=True)

    print 'T_p, T_iou', T_p, T_iou
    print 'test size', len(validset)

    os.system('rm %s/*' % outdir)
    with torch.no_grad():
        cls_out = valid(outdir, net, T_p, T_iou, validset, device, drawimg, debug)
        imgids = zip(*validset.imgnames)[-1]
        cpu_ap(outdir, cls_out, imgids, validroot)

def cpu_ap(detpath, cls_out, imgids, validroot):
    classes = conf.init_classes.classmap.keys()

    annopath = '%s/Annotations/' % validroot + '{:s}.xml'

    aps = []

    for _, cls in enumerate(classes):
        if cls not in cls_out:
            continue

        filename = os.path.join(detpath, cls)
        if not os.path.exists(filename):
            continue

        rec, prec, ap = voc_eval.voc_eval(filename, annopath, imgids, cls, cachedir=None, ovthresh=0.5)
        aps += [ap]

        print '#', cls
        print 'AP {:.4f}'.format(ap)
        print 'recall {:.4f}'.format(rec[-1])
        print 'precision {:.4f}'.format(prec[-1])
        print

    print '-' * 10
    print 'mAP {:.4f}'.format(np.mean(aps))

if __name__ == "__main__":
    fire.Fire(main)
