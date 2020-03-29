# train
python main.py --cuda=cuda:1 --epoch=180 --eta=0.0001 --bat=16 --codename=voc --models=darknet53 --summary=True --jitterimg=True --shift=True

# image detect
python valid.py --checkpoint=checkpoint/v3.last_init_voc.end1_darknet53 --maxsize=100 --drawimg=True --debug=True --models=darknet53 --T_p=0.35

# video detect
python cvideo.py --mp4=video/379489.mp4 --T_p=0.35 --checkpoint=checkpoint/v3.last_init_voc.end1_darknet53
