import time
import cv2
import numpy as np
import glob
import os
import renom as rm
from renom.optimizer import Sgd
from renom.utility.trainer import Trainer
from renom.cuda.cuda import set_cuda_active
from renom import cuda
from renom.utility.distributor import NdarrayDistributor
from yolov2 import *
from utils import *

# set_cuda_active(True)

# hyper parameters
backup_path = "./weights"
yolo_weight_file = "%s/yolov2_final.h5" % (backup_path)

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005
classes = 20
bbox = 5
detection_thresh = 0.02
iou_thresh = 0.3

# load image generator
print("loading image generator...")


# load model
print("loading initial model...")

pretrained_model = Pretrained(classes=1000)

weights_path = "./weights/darknet19_448.weights"

with open(weights_path, "rb") as f:
    dat = np.fromfile(f, dtype=np.float32)[4:]
    offset = 0
    layers = [[3,32,3],[32,64,3],[64,128,3],[128, 64,1],[64,128,3],[128,256,3],[256,128,1],[128,256,3],
        [256,512,3],[512,256,1],[256,512,3],[512,256,1],[256,512,3],[512,1024,3],[1024,512,1],
        [512,1024,3],[1024,512,1],[512,1024,3]]

    for i, l in enumerate(layers):
        in_ch = l[0]
        out_ch = l[1]
        size = l[2]


        txt = "pretrained_model.conv%d.beta = rm.Variable(dat[%d:%d].reshape(1, %d, 1, 1))" % (i+1, offset, offset+out_ch, out_ch)
        offset += out_ch
        exec(txt)

        #TODO batch_normalizeのパラメタ
        txt = "pretrained_model.conv%d.gamma = rm.Variable(dat[%d:%d].reshape(1, %d, 1, 1))" % (i+1, offset, offset+out_ch, out_ch)
        offset += out_ch
        exec(txt)

        txt = "pretrained_model.conv%d.bn._mov_mean = rm.Variable(dat[%d:%d])" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt = "pretrained_model.conv%d.bn._mov_std = rm.Variable(np.sqrt(dat[%d:%d]))" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt = "pretrained_model.conv%d.conv.params['w'] = rm.Variable(dat[%d:%d].reshape(%d,%d,%d,%d))" % (i+1, offset, offset+(out_ch*in_ch*size*size), out_ch, in_ch, size, size)
        offset += out_ch*in_ch*size*size
        exec(txt)

        txt = "pretrained_model.conv%d.conv.params['b'] = rm.Variable(np.zeros((1, %d, 1, 1)))" % (i+1, out_ch)
        exec(txt)

        print(i, offset)

    in_ch = 1024
    out_ch = 1000

    txt= "pretrained_model.conv23.params['b'] = rm.Variable(dat[%d:%d])" % (offset, offset+out_ch)
    offset += out_ch
    exec(txt)

    txt= "pretrained_model.conv23.params['w'] = rm.Variable(dat[%d:%d].reshape(%d,%d,1,1))" % (offset, offset+out_ch*in_ch*1, out_ch,in_ch)
    offset += out_ch*in_ch*1
    exec(txt)

    print(dat.shape[0] - offset)


yolo_model = YOLOv2(classes=classes, bbox=bbox)
yolo_model.load(yolo_weight_file)
voc = VOCgenerator()

print("start testing")

image_generator = voc.generate_samples(batch_size=1, train=False)
img, t = next(image_generator)
print(img[0].shape)
input_width = img[0].shape[0]
input_height = img[0].shape[1]
x, y, w, h, conf, prob = yolo_predict(yolo_model, pretrained_model, img)
_, _, _, grid_h, grid_w = x.shape
x = np.reshape(x, (bbox, grid_h, grid_w))
y = np.reshape(y, (bbox, grid_h, grid_w))
w = np.reshape(w, (bbox, grid_h, grid_w))
h = np.reshape(h, (bbox, grid_h, grid_w))
conf = np.reshape(conf, (bbox, grid_h, grid_w))
prob = np.transpose(np.reshape(prob, (bbox, classes, grid_h, grid_w)), (1, 0, 2, 3))
detected_indices = (conf * prob).max(axis=0) > detection_thresh

results = []
for i in range(detected_indices.sum()):
    results.append({
        "label": prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax(),
        "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
        "conf" : conf[detected_indices][i],
        "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
        "box"  : Box(
                    x[detected_indices][i]*input_width,
                    y[detected_indices][i]*input_height,
                    w[detected_indices][i]*input_width,
                    h[detected_indices][i]*input_height).crop_region(input_height, input_width)
    })

# nms
print(results)
nms_results = nms(results, iou_thresh)
