import torch

usevoc = True
if usevoc:
    from base import init_classes20 as init_classes
else:
    from base import init_classes80 as init_classes

gk_noobj, gk_obj, gk_coor, gk_class = 'noobj', 'obj', 'coor', 'class'

loss_method = 'sum'
gmap = {}
gmap[gk_noobj] = 0.5
gmap[gk_obj] = 1.5
gmap[gk_coor] = 1.0
gmap[gk_class] = 1.0

colormap = []
for i in range(0, 255, 55):
    for j in range(0, 255, 55):
        for k in range(0, 255, 55):
            colormap.append((i, j, k))

trainroot = ['vocdata/voc2012', 'vocdata/voc2007_train']
validroot = ['vocdata/voc2007_val']
eta_decay = {5:1e-3, 120:0.1, 200:0.01}
anchors_tup = (1.32, 1.73), (3.19, 4.0), (5.05, 8.09), (9.47, 4.84), (11.23, 10.0)
anchors = torch.zeros(len(anchors_tup), 2)
for i, it in enumerate(anchors_tup):
    anchors[i][0], anchors[i][1] = it

loc_box = 0
loc_p = 4
loc_cls = 5
class_num = max(init_classes.classmap.values()) + 1
class_cross = True
seedmap = {14:0.1, 6:0.5, 8:0.4, 4:0.6, 10:0.7, 7:0.8, 2:0.8}
coco_maxobj = 6

gmode = torch.tensor([11, 13, 15, 17]).repeat(32, 1).t().contiguous().view(-1).flip(0)
v_resize = 32 * 13
