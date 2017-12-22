#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
from multiprocessing import Pipe, Process

from utils import *

import renom as rm
from renom.cuda import cuda
from renom.optimizer import Sgd, Adam
from renom.core import DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH

from yolo_detector import *

class YOLOv2(rm.Model):

    def __init__(self, classes, bbox):
        super(YOLOv2, self).__init__()

        self.bbox = bbox
        self.classes = classes
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]
        self.thresh = 0.6

            ##### common layers for both pretrained layers and yolov2 #####
        self.conv1  = rm.Conv2d(channel=32, filter=3, stride=1, padding=1)
        self.bn1 = rm.BatchNormalize(mode='feature')
        self.pool1 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv2  = rm.Conv2d(channel=64, filter=3, stride=1, padding=1)
        self.bn2 = rm.BatchNormalize(mode='feature')
        self.pool2 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv3  = rm.Conv2d(channel=128, filter=3, stride=1, padding=1)
        self.bn3 = rm.BatchNormalize(mode='feature')
        self.conv4  = rm.Conv2d(channel=64, filter=1, stride=1, padding=0)
        self.bn4 = rm.BatchNormalize(mode='feature')
        self.conv5  = rm.Conv2d(channel=128, filter=3, stride=1, padding=1)
        self.bn5 = rm.BatchNormalize(mode='feature')
        self.pool3 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv6  = rm.Conv2d(channel=256, filter=3, stride=1, padding=1)
        self.bn6 = rm.BatchNormalize(mode='feature')
        self.conv7  = rm.Conv2d(channel=128, filter=1, stride=1, padding=0)
        self.bn7 = rm.BatchNormalize(mode='feature')
        self.conv8  = rm.Conv2d(channel=256, filter=3, stride=1, padding=1)
        self.bn8 = rm.BatchNormalize(mode='feature')
        self.pool4 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv9  = rm.Conv2d(channel=512, filter=3, stride=1, padding=1)
        self.bn9 = rm.BatchNormalize(mode='feature')
        self.conv10  = rm.Conv2d(channel=256, filter=1, stride=1, padding=0)
        self.bn10 = rm.BatchNormalize(mode='feature')
        self.conv11  = rm.Conv2d(channel=512, filter=3, stride=1, padding=1)
        self.bn11 = rm.BatchNormalize(mode='feature')
        self.conv12  = rm.Conv2d(channel=256, filter=1, stride=1, padding=0)
        self.bn12 = rm.BatchNormalize(mode='feature')
        self.conv13  = rm.Conv2d(channel=512, filter=3, stride=1, padding=1)
        self.bn13 = rm.BatchNormalize(mode='feature')
        self.pool5 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv14  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn14 = rm.BatchNormalize(mode='feature')
        self.conv15  = rm.Conv2d(channel=512, filter=1, stride=1, padding=0)
        self.bn15 = rm.BatchNormalize(mode='feature')
        self.conv16  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn16 = rm.BatchNormalize(mode='feature')
        self.conv17  = rm.Conv2d(channel=512, filter=1, stride=1, padding=0)
        self.bn17 = rm.BatchNormalize(mode='feature')
        self.conv18  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn18 = rm.BatchNormalize(mode='feature')

        ###### pretraining layer
        self.conv23 = rm.Conv2d(channel=classes, filter=1, stride=1, padding=0)

        ###### detection layer
        self.conv19  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn19 = rm.BatchNormalize(mode='feature')
        self.conv20  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn20 = rm.BatchNormalize(mode='feature')
        self.conv21  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1) ##input=3072
        self.bn21 = rm.BatchNormalize(mode='feature')
        self.conv22  = rm.Conv2d(channel=bbox * (5 + classes), filter=1, stride=1, padding=0)




    def forward(self, x):
        h = self.pool1(rm.leaky_relu(self.bn1(self.conv1(x)), slope=0.1))
        h = self.pool2(rm.leaky_relu(self.bn2(self.conv2(h)), slope=0.1))
        h = rm.leaky_relu(self.bn3(self.conv3(h)), slope=0.1)
        h = rm.leaky_relu(self.bn4(self.conv4(h)), slope=0.1)
        h = self.pool3(rm.leaky_relu(self.bn5(self.conv5(h)), slope=0.1))
        h = rm.leaky_relu(self.bn6(self.conv6(h)), slope=0.1)
        h = rm.leaky_relu(self.bn7(self.conv7(h)), slope=0.1)
        h = self.pool4(rm.leaky_relu(self.bn8(self.conv8(h)), slope=0.1))
        h = rm.leaky_relu(self.bn9(self.conv9(h)), slope=0.1)
        h = rm.leaky_relu(self.bn10(self.conv10(h)), slope=0.1)
        h = rm.leaky_relu(self.bn11(self.conv11(h)), slope=0.1)
        h = rm.leaky_relu(self.bn12(self.conv12(h)), slope=0.1)
        h = rm.leaky_relu(self.bn13(self.conv13(h)), slope=0.1)
        high_resolution_feature = reorg(h) # 高解像度特徴量をreorgでサイズ落として保存しておく
        h = self.pool5(h)
        h = rm.leaky_relu(self.bn14(self.conv14(h)), slope=0.1)
        h = rm.leaky_relu(self.bn15(self.conv15(h)), slope=0.1)
        h = rm.leaky_relu(self.bn16(self.conv16(h)), slope=0.1)
        h = rm.leaky_relu(self.bn17(self.conv17(h)), slope=0.1)
        h = rm.leaky_relu(self.bn18(self.conv18(h)), slope=0.1)

        ##### detection layer
        h = rm.leaky_relu(self.bn19(self.conv19(h)), slope=0.1)
        h = rm.leaky_relu(self.bn20(self.conv20(h)), slope=0.1)
        h = rm.concat(high_resolution_feature, h)
        h = rm.leaky_relu(self.bn21(self.conv21(h)), slope=0.1)
        h = self.conv22(h)

        return h

    def init_anchors(self, anchors):
        self.anchors = anchors




#
# class YOLOv2Predictor(rm.Model):
#
#     def __init__(self, predictor):
#         super(YOLOv2Predictor, self).__init__(predictor=predictor)
#         self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]
#         self.thresh = 0.6
#         # self.seen = 0
#         # self.unstable_seen = 5000
#
#     def init_anchors(self, anchors):
#         self.anchors = anchors


def yolo_train(model, input_x, t, opt, weight_decay):
    with model.train():
        output = model(input_x)

        wd = 0
        for i in range(1, 23):
            m = eval("model.conv%d" % i)
            if hasattr(m, "params"):
                w = m.params.get("w", None)
                if w is not None:
                    wd += rm.sum(w**2)

        loss = yolo_detector(output, t, bbox=model.bbox, classes=model.classes, init_anchors=model.anchors) + weight_decay * wd

    grad = loss.grad()
    grad.update(opt)
    return loss

def yolo_predict(model, input_x):
    output = model(input_x)
    batch_size, _, grid_h, grid_w = output.shape
    output_reshape = np.reshape(output, (batch_size, model.bbox, model.classes+5, grid_h, grid_w))
    x, y, w, h, conf, prob = output_reshape[:,:,0:1,:,:], output_reshape[:,:,1:2,:,:],output_reshape[:,:,2:3,:,:], output_reshape[:,:,3:4,:,:], output_reshape[:,:,4:5,:,:], output_reshape[:,:,5:,:,:]
    x = rm.sigmoid(x) # xのactivation
    y = rm.sigmoid(y) # yのactivation
    conf = rm.sigmoid(conf) # confのactivation
    prob = np.transpose(prob, (0, 2, 1, 3, 4))
    prob = rm.softmax(prob) # probablitiyのacitivation
    prob = np.transpose(prob, (0, 2, 1, 3, 4))

    # x, y, w, hを絶対座標へ変換
    x_shift = np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape)
    y_shift = np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape)
    w_anchor = np.broadcast_to(np.reshape(np.array(model.anchors, dtype=np.float32)[:, 0], (model.bbox, 1, 1, 1)), w.shape)
    h_anchor = np.broadcast_to(np.reshape(np.array(model.anchors, dtype=np.float32)[:, 1], (model.bbox, 1, 1, 1)), h.shape)
    #x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
    box_x = (x + x_shift) / grid_w
    box_y = (y + y_shift) / grid_h
    box_w = np.exp(w) * w_anchor / grid_w
    box_h = np.exp(h) * h_anchor / grid_h

    return box_x, box_y, box_w, box_h, conf, prob
