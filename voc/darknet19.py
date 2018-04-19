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


class Darknet19(rm.Model):

    def __init__(self, classes):
        super(Darknet19, self).__init__()
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
        h = self.pool5(h)
        h = rm.leaky_relu(self.bn14(self.conv14(h)), slope=0.1)
        h = rm.leaky_relu(self.bn15(self.conv15(h)), slope=0.1)
        h = rm.leaky_relu(self.bn16(self.conv16(h)), slope=0.1)
        h = rm.leaky_relu(self.bn17(self.conv17(h)), slope=0.1)
        h = rm.leaky_relu(self.bn18(self.conv18(h)), slope=0.1)

        ##### pretraining layer
        h = self.conv23(h)
        h = rm.average_pool2d(h, filter=(h.shape[-1], h.shape[-1]), stride=(1, 1), padding=(0, 0))

        y = rm.reshape(h, (x.shape[0], -1))

        return y

class Darknet19Predictor(rm.Model):
    def __init__(self, predictor):
        super(Darknet19Predictor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)

        # if t.ndim == 2: # use squared error when label is one hot label
        #     y = rm.softmax(y)
        #     # loss = F.mean_squared_error(y, t)
        #     loss = rm.mean_squared_error(y, t)
        #     accuracy = rm.accuracy(y, t.argmax(axis=1).astype(np.int32))
        # else: # use softmax cross entropy when label is normal label
        loss = rm.softmax_cross_entropy(y, t)

        return loss

    def predict(self, x):
        y = self.predictor(x)
        return rm.softmax(y)
