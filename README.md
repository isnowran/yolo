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
```
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/v2_voc_411907.gif)

## yolo_v2 and vocdata, image detect
`python valid.py --checkpoint=checkpoint/v2.resnet152.dump --maxsize=100 --drawimg=True --debug=False --models=resnet152 --T_p=0.35`

![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/24_007339.jpg)
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/28_007237.jpg)
![image](https://github.com/isnowran/yolo/blob/master/demo.v2_voc.detect.jpg/31_000467.jpg)

## train v2/v3, voc/coco, resnet/darknet
`python main.py --cuda=cuda:1 --epoch=180 --eta=0.0001 --bat=16 --codename=voc --models=darknet53 --summary=True --jitterimg=True --shift=True`

## view loss
`tensorboard --logdir runs --bind_all`
