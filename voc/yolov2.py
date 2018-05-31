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

class DarknetConv2d(object):

    def __init__(self, channel, filter, stride, padding):
        self.channel = channel
        self.conv = rm.Conv2d(channel=channel, filter=filter, padding=padding, stride=stride)
        self.bn = rm.BatchNormalize(mode="feature")
        self.bn.inference = True
        self.gamma = rm.Variable(np.ones((1, channel, 1, 1)))
        self.beta = rm.Variable(np.zeros((1, channel, 1, 1)))

    def __call__(self, x):
        return self.gamma * self.bn(self.conv(x)) + self.beta





class Pretrained(rm.Model):

    def __init__(self, classes):
        super(Pretrained, self).__init__()
            ##### common layers for both pretrained layers and yolov2 #####
        self.conv1  = DarknetConv2d(channel=32, filter=3, stride=1, padding=1)
        self.pool1 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv2  = DarknetConv2d(channel=64, filter=3, stride=1, padding=1)
        self.pool2 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv3  = DarknetConv2d(channel=128, filter=3, stride=1, padding=1)
        self.conv4  = DarknetConv2d(channel=64, filter=1, stride=1, padding=0)
        self.conv5  = DarknetConv2d(channel=128, filter=3, stride=1, padding=1)
        self.pool3 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv6  = DarknetConv2d(channel=256, filter=3, stride=1, padding=1)
        self.conv7  = DarknetConv2d(channel=128, filter=1, stride=1, padding=0)
        self.conv8  = DarknetConv2d(channel=256, filter=3, stride=1, padding=1)
        self.pool4 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv9  = DarknetConv2d(channel=512, filter=3, stride=1, padding=1)
        self.conv10  = DarknetConv2d(channel=256, filter=1, stride=1, padding=0)
        self.conv11  = DarknetConv2d(channel=512, filter=3, stride=1, padding=1)
        self.conv12  = DarknetConv2d(channel=256, filter=1, stride=1, padding=0)
        self.conv13  = DarknetConv2d(channel=512, filter=3, stride=1, padding=1)
        self.pool5 = rm.MaxPool2d(filter=2, stride=2, padding=0)
        self.conv14  = DarknetConv2d(channel=1024, filter=3, stride=1, padding=1)
        self.conv15  = DarknetConv2d(channel=512, filter=1, stride=1, padding=0)
        self.conv16  = DarknetConv2d(channel=1024, filter=3, stride=1, padding=1)
        self.conv17  = DarknetConv2d(channel=512, filter=1, stride=1, padding=0)
        self.conv18  = DarknetConv2d(channel=1024, filter=3, stride=1, padding=1)

        ###### pretraining layer
        self.conv23 = rm.Conv2d(channel=classes, filter=1, stride=1, padding=0)

    def forward(self, x):
        h = self.pool1(rm.leaky_relu(self.conv1(x), slope=0.1))
        h = self.pool2(rm.leaky_relu(self.conv2(h), slope=0.1))
        h = rm.leaky_relu(self.conv3(h), slope=0.1)
        h = rm.leaky_relu(self.conv4(h), slope=0.1)
        h = self.pool3(rm.leaky_relu(self.conv5(h), slope=0.1))
        h = rm.leaky_relu(self.conv6(h), slope=0.1)
        h = rm.leaky_relu(self.conv7(h), slope=0.1)
        h = self.pool4(rm.leaky_relu(self.conv8(h), slope=0.1))
        h = rm.leaky_relu(self.conv9(h), slope=0.1)
        h = rm.leaky_relu(self.conv10(h), slope=0.1)
        h = rm.leaky_relu(self.conv11(h), slope=0.1)
        h = rm.leaky_relu(self.conv12(h), slope=0.1)
        h = rm.leaky_relu(self.conv13(h), slope=0.1)
        high_resolution_feature = reorg(h)
        h = self.pool5(h)
        h = rm.leaky_relu(self.conv14(h), slope=0.1)
        h = rm.leaky_relu(self.conv15(h), slope=0.1)
        h = rm.leaky_relu(self.conv16(h), slope=0.1)
        h = rm.leaky_relu(self.conv17(h), slope=0.1)
        h = rm.leaky_relu(self.conv18(h), slope=0.1)

        return h, high_resolution_feature



class YOLOv2(rm.Model):

    def __init__(self, classes, bbox):
        super(YOLOv2, self).__init__()

        self.bbox = bbox
        self.classes = classes
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]
        self.thresh = 0.6

        ###### detection layer
        self.conv19  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn19 = rm.BatchNormalize(mode='feature')
        self.conv20  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1)
        self.bn20 = rm.BatchNormalize(mode='feature')
        self.conv21  = rm.Conv2d(channel=1024, filter=3, stride=1, padding=1) ##input=3072
        self.bn21 = rm.BatchNormalize(mode='feature')
        self.conv22  = rm.Conv2d(channel=bbox * (5 + classes), filter=1, stride=1, padding=0)

    def forward(self, x, high_resolution_feature):
        ##### detection layer
        h = rm.leaky_relu(self.bn19(self.conv19(x)), slope=0.1)
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


def yolo_train(yolo_model, pretrained_model, input_x, t, opt, weight_decay):
    pretrained_model.set_models(inference=True)
    yolo_model.set_models(inference=False)
    with yolo_model.train():
        pretrained_x = pretrained_model(input_x)
        x = pretrained_x[0].as_ndarray()
        feature = pretrained_x[1].as_ndarray()
        output = yolo_model(x, feature)

        wd = 0
        for i in range(19, 23):
            m = eval("yolo_model.conv%d" % i)
            if hasattr(m, "params"):
                w = m.params.get("w", None)
                if w is not None:
                    wd += rm.sum(w**2)
        loss = yolo_detector(output, t, bbox=yolo_model.bbox, classes=yolo_model.classes, init_anchors=yolo_model.anchors) + weight_decay * wd
    grad = loss.grad()
    grad.update(opt)
    return loss

def yolo_predict(yolo_model, pretrained_model, input_x):
    pretrained_x = pretrained_model(input_x)
    x = pretrained_x[0].as_ndarray()
    feature = pretrained_x[1].as_ndarray()
    output = yolo_model(x, feature)
    batch_size, _, grid_h, grid_w = output.shape
    output_reshape = np.reshape(output, (batch_size, yolo_model.bbox, yolo_model.classes+5, grid_h, grid_w))
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
    w_anchor = np.broadcast_to(np.reshape(np.array(yolo_model.anchors, dtype=np.float32)[:, 0], (yolo_model.bbox, 1, 1, 1)), w.shape)
    h_anchor = np.broadcast_to(np.reshape(np.array(yolo_model.anchors, dtype=np.float32)[:, 1], (yolo_model.bbox, 1, 1, 1)), h.shape)
    #x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
    box_x = (x + x_shift) / grid_w
    box_y = (y + y_shift) / grid_h
    box_w = np.exp(w) * w_anchor / grid_w
    box_h = np.exp(h) * h_anchor / grid_h

    return box_x, box_y, box_w, box_h, conf, prob
