## yolo_v2 and vocdata, video detect
```
[binary@Xiaobai v2.src]$ . merge_dump.sh
[binary@Xiaobai v2.src]$ ls checkpoint
v2.resnet152.dump
[binary@Xiaobai v2.src]$ python cvideo.py --mp4=video/379489.mp4 --T_p=0.35 --checkpoint=checkpoint/v2.resnet152.dump --models=resnet152
load checkpoint checkpoint/v2.resnet152.dump
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
1 fps 0.61 0.71 x 0.92 = 0.65 boxes (42, 1, 898, 539) motorbike
2 fps 0.62 0.71 x 0.92 = 0.65 boxes (41, 1, 900, 539) motorbike
3 fps 0.62 0.70 x 0.92 = 0.65 boxes (40, 1, 901, 539) motorbike
...
```
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/v2_voc_411907.gif)

## yolo_v2 and vocdata, image detect
```
[binary@Xiaobai v2.src]$ mkdir img_out output
[binary@Xiaobai v2.src]$ python valid.py --checkpoint=checkpoint/v2.resnet152.dump --maxsize=100 --drawimg=True --debug=False --models=resnet152 --T_p=0.35 --maxsize=5
load checkpoint checkpoint/v2.resnet152.dump
T_p, T_iou 0.35 0.5
test size 5
100%|██████████| 5/5 [00:08<00:00,  1.70s/it]
# aeroplane
AP 1.0000
recall 1.0000
precision 1.0000

...

# tvmonitor
AP 1.0000
recall 1.0000
precision 1.0000

----------
mAP 0.7667
[binary@Xiaobai v2.src]$
```

![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/24_007339.jpg)
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/28_007237.jpg)
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/31_000467.jpg)

## train v2/v3, voc/coco, resnet/darknet
```
[binary@Xiaobai v2.src]$ python main.py --cuda=cuda:1 --epoch=180 --eta=0.0001 --bat=16 --codename=voc --models=darknet53 --summary=True --jitterimg=True --shift=True
device: cpu cpu
models: darknet53
jitterimg True
shift True
optim: SGD
init trainset...
train_size:19626
init validset...
valid_size:2510
  0%|          | 0/180 [00:00<?, ?it/s]
```
## view loss
```
binary@jita:~/v3.src$ tensorboard --logdir runs --bind_all
TensorBoard 2.2.0 at http://jita:6006/ (Press CTRL+C to quit)
```
