import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import caffe
from lib import run_net
from lib import score_util

from datasets.cityscapes import cityscapes
random.seed(0xCAFFE)

# Configure caffe and load net
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('./prototxt/stage-cityscapes-fcn8s.prototxt',
                '/data/syzhang/clockwork-nets/cityscapes-fcn8s-heavy.caffemodel',
                caffe.TEST)

# Dataset details
CS = cityscapes('/data/seg_dataset/')
n_cl = len(CS.classes)
split = 'val'
label_frames = CS.list_label_frames(split)

# Oracle per frame
hist_perframe = np.zeros((n_cl, n_cl))
for i, idx in enumerate(label_frames):
    if i % 100 == 0:
        print 'running {}/{}'.format(i, len(label_frames))
    city = idx.split('_')[0]
    # idx is city_shot_frame
    im = CS.load_image(split, city, idx)
    out = run_net.segrun(net, CS.preprocess(im))
    label = CS.load_label(split, city, idx)
    hist_perframe += score_util.fast_hist(label.flatten(), out.flatten(), n_cl)

accP, cl_accP, mean_iuP, fw_iuP = score_util.get_scores(hist_perframe)
print 'Oracle: Per frame'
print 'acc\t\t cl acc\t\t mIU\t\t fwIU'
print '{:f}\t {:f}\t {:f}\t {:f}\t'.format(100*accP, 100*cl_accP, 100*mean_iuP, 100*fw_iuP)
