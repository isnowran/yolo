import sys
sys.path.append("../")

import os
import torch
from tqdm import trange
import numpy as np
import fire
import conf

import loader
from base.dnet53 import darknet53_v3 as darknet53
from base.snet import s_resnet
from base import libiou
from base import libnet
from base import util
from thirdlib import voc_eval

def writerec(outdir, cmap, cname):
    for k, vs in cmap.iteritems():
        cname.add(k)

        with open("%s/%s" % (outdir, k), 'a+') as fd:
            for v in vs:
                fd.write('%s %.3f %d %d %d %d\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))

def ss(x):
    return x.detach().cpu().numpy().round(4)

def valid(outdir, net, T_p, T_iou, validset, device, drawimg=False, debug=False):
    cmap = {}
    cset = set()

    for i in trange(len(validset)):
        imgx_ori, imgx, cinfo, objs, imgid = validset[i]
        imgx_t = imgx.unsqueeze(0).to(device)

        with torch.no_grad():
            rss = [libnet.refine(r, conf.class_num+5) for r in net(imgx_t)]

        # ap
        h, w = imgx_ori.shape[-2:]

        box_all = []
        ps = []
        recs = []
        cs = []
        cinds = []
        for rsind, rs in enumerate(rss):
            rs = rs.squeeze()
            p, rec, maxv, maxind = util.create_box(rs, (w, h), conf.grids[rsind], conf.anchor_split[rsind].to(device))

            if debug:
                for _y, it0 in enumerate(p):
                    for _x, its in enumerate(it0):
                        key = _y, _x

                        for an_indx, _ in enumerate(its):
                            box_all.append((p[_y, _x, an_indx], rec[_y, _x, an_indx], key, maxind[_y, _x, an_indx], maxv[_y, _x, an_indx], conf.grids[rsind], an_indx))

            ps.append(p.reshape(-1))
            recs.append(rec.view(-1, rec.shape[-1]))
            cs.append(maxv.reshape(-1))
            cinds.append(maxind.reshape(-1))

        if debug:
            print 'proc', imgid

            has_objs = []
            for points, cell, best_rsind, best_anchorind in cinfo:
                key = tuple(cell.tolist())
                has_objs.append(key)

            for p, box, cellID, cind, cval, _grid, _anchor in sorted(box_all, key=lambda x: x[0] * x[4])[-30:]:
                print 'p: %.2f x %.2f = %.2f' % (p.detach(), cval.detach(), p.detach() * cval.detach()), 'rec', box.detach().cpu(), 'class', cind.detach().cpu(), 'cell', cellID[::-1], 'grid', _grid, 'ancor', _anchor

            for ee, obj in enumerate(objs):
                points, cell, best_rsind, best_anchorind = cinfo[ee]
                box = libiou.cent2rec((w, h), conf.grids[best_rsind], cell, points)
                print '-- class', int(objs[ee]), (best_rsind.tolist(), best_anchorind.tolist()), box[2:] - box[:2], box, 'cell', cell.tolist()

        box_pure = util.nms(ps, recs, cs, cinds, T_p, T_iou)

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

def main(T_p=0.35, T_iou=0.5, checkpoint=None, models=None, validroot='vocdata/voc2007_test', cuda='cpu', maxsize=0, drawimg=False, debug=False):
    outdir = 'output'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    if not models:
        models = util.extact_model(checkpoint)

    if models == 'darknet53':
        net = darknet53(conf.class_num)
    else:
        net = s_resnet(models, False, conf.class_num)

    net = net.to(device)
    net.eval()

    if checkpoint:
        print 'load checkpoint', checkpoint
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    validset = loader.imgloader([validroot], conf.input_size, jitterimg=False, shift=False, maxsize=maxsize, cache_ori=True)

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
