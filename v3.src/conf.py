import torch

usevoc = True
if usevoc:
    from base import init_classes20 as init_classes
else:
    from base import init_classes80 as init_classes

gk_noobj, gk_obj, gk_coor, gk_class = 'noobj', 'obj', 'coor', 'class'

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

trainroot = ['vocdata/voc2007_train', 'vocdata/voc2012']
validroot = ['vocdata/voc2007_val']

eta_decay = {2:1e-3, 60:0.1, 90:0.01}

input_size = 416
grids = 13, 26, 52
units = [input_size / g for g in grids]

class_num = max(init_classes.classmap.values()) + 1
'''
_anchor52 = torch.tensor([[10, 13], [16, 30], [33, 23]]).float()
_anchor26 = torch.tensor([[30, 61], [62, 45], [59, 119]]).float()
_anchor13 = torch.tensor([[116, 90], [156, 198], [373, 326]]).float()
'''
if class_num == 20:
    _anchor13 = torch.tensor([[341, 356], [164, 307], [235, 176]])
    _anchor26 = torch.tensor([[89, 212], [106, 101], [48, 127]])
    _anchor52 = torch.tensor([[52, 44], [24, 67], [14, 25]])
elif class_num == 90:
    _anchor13 = torch.tensor([[282, 293], [111, 174], [85, 65]])
    _anchor26 = torch.tensor([[41, 106], [40, 33], [19, 52]])
    _anchor52 = torch.tensor([[20, 15], [9, 26], [5, 9]])

anchor_split = [an.float() / units[i] for (i, an) in enumerate([_anchor13, _anchor26, _anchor52])]
anchors = torch.cat((_anchor13, _anchor26, _anchor52))
best_anchor_threshold = 0.6

loc_box = 0
loc_p = 4
loc_cls = 5

# coco about
# seedmap = {14:0.1, 6:0.5, 8:0.4, 4:0.6, 10:0.7, 7:0.8, 2:0.8}
seedmap = {}

#coco_maxobj = 6
coco_maxobj = 100
