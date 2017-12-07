#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import renom as rm
from utils import *
from renom.core import Node, get_gpu, to_value
from renom.cuda import cuda as cu


class yolo_detector(Node):

    def __new__(cls, output, t, bbox, classes, init_anchors):
       # assert rhs.ndim == 4, "Input arrays must have 4 dimenstions."
        return cls.calc_value(output, t, bbox, classes, init_anchors)


    @classmethod
    def _oper_cpu(cls, output, t, bbox, classes, init_anchors):
        batch_size, _, grid_h, grid_w = output.shape
        output_reshape = rm.reshape(output, (batch_size, bbox, classes+5, grid_h, grid_w))
        x, y, w, h, conf, prob = output_reshape[:,:,0:1,:,:], output_reshape[:,:,1:2,:,:],output_reshape[:,:,2:3,:,:], output_reshape[:,:,3:4,:,:], output_reshape[:,:,4:5,:,:], output_reshape[:,:,5:,:,:]
        x = rm.sigmoid(x)
        y = rm.sigmoid(y)
        conf = rm.sigmoid(conf)
        prob = np.transpose(prob, (0, 2, 1, 3, 4)).reshape(batch_size, classes, -1)
        prob = rm.softmax(prob)
        # prob_exp = np.exp(prob)
        # prob = prob_exp / np.sum(prob_exp, axis=1, keepdims=True)
        prob = rm.reshape(prob, (batch_size, classes, bbox, grid_h, grid_w))
        deltas = np.zeros(output_reshape.shape)
        #anchor
        if init_anchors is None:
            anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]
        else:
            anchors = init_anchors

        thresh = 0.6
        # 教師データ
        tw = np.zeros(w.shape)
        th = np.zeros(h.shape)
        tx = np.tile(0.5, x.shape).astype(np.float32)
        ty = np.tile(0.5, y.shape).astype(np.float32)
        box_learning_scale = np.tile(0.1, x.shape).astype(np.float32)

        tconf = np.zeros(conf.shape, dtype=np.float32)
        conf_learning_scale = np.tile(0.1, conf.shape).astype(np.float32)

        tprob = prob.as_ndarray()
        print("output")
        print(output_reshape[1,1, :,1,1])
        x_shift = np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape[1:])
        y_shift = np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape[1:])
        w_anchor = np.broadcast_to(np.reshape(np.array(anchors, dtype=np.float32)[:, 0], (bbox, 1, 1, 1)), w.shape[1:])
        h_anchor = np.broadcast_to(np.reshape(np.array(anchors, dtype=np.float32)[:, 1], (bbox, 1, 1, 1)), h.shape[1:])
        #x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()

        best_ious = []
        for batch in range(batch_size):
            truth_bbox = len(t[batch])
            box_x = (x[batch] + x_shift) / grid_w
            box_y = (y[batch] + y_shift) / grid_h
            box_w = np.exp(w[batch]) * w_anchor / grid_w
            box_h = np.exp(h[batch]) * h_anchor / grid_h
            ious = []
            for truth_index in range(truth_bbox):
                truth_box_x = np.broadcast_to(np.array(t[batch][truth_index]["x"], dtype=np.float32), box_x.shape)
                truth_box_y = np.broadcast_to(np.array(t[batch][truth_index]["y"], dtype=np.float32), box_y.shape)
                truth_box_w = np.broadcast_to(np.array(t[batch][truth_index]["w"], dtype=np.float32), box_w.shape)
                truth_box_h = np.broadcast_to(np.array(t[batch][truth_index]["h"], dtype=np.float32), box_h.shape)
                #truth_box_x.to_gpu(), truth_box_y.to_gpu(), truth_box_w.to_gpu(), truth_box_h.to_gpu()
                ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h), Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)))
            ious = np.array(ious)
            best_ious.append(np.max(ious, axis=0))
        best_ious = np.array(best_ious)
        tconf[best_ious > thresh] = conf[best_ious > thresh]
        conf_learning_scale[best_ious > thresh] = 0

        abs_anchors = anchors / np.array([grid_w, grid_h])
        for batch in range(batch_size):
            for truth_box in t[batch]:
                truth_h = int(float(truth_box["x"]) * grid_w)
                truth_w = int(float(truth_box["y"]) * grid_h)
                truth_n = 0
                best_iou = 0.0
                for anchor_index, abs_anchor in enumerate(abs_anchors):
                    iou = box_iou(Box(0, 0, float(truth_box["w"]), float(truth_box["h"])), Box(0, 0, abs_anchor[0], abs_anchor[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_n = anchor_index

                box_learning_scale[batch, truth_n, :, truth_h, truth_w] = 1.0
                tx[batch, truth_n, :, truth_h, truth_w] = float(truth_box["x"]) * grid_w - truth_w
                ty[batch, truth_n, :, truth_h, truth_w] = float(truth_box["y"]) * grid_h - truth_h
                tw[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box["w"]) / abs_anchors[truth_n][0])
                th[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box["h"]) / abs_anchors[truth_n][1])
                tprob[batch, :, truth_n, truth_h, truth_w] = 0
                tprob[batch, int(truth_box["label"]), truth_n, truth_h, truth_w] = 1

                full_truth_box = Box(float(truth_box["x"]), float(truth_box["y"]), float(truth_box["w"]), float(truth_box["h"]))
                predicted_box = Box(
                    (x[batch, truth_n, 0, truth_h, truth_w] + truth_w) / grid_w,
                    (y[batch, truth_n, 0, truth_h, truth_w] + truth_h) / grid_h,
                    np.exp(w[batch, truth_n, 0, truth_h, truth_w]) * abs_anchors[truth_n][0],
                    np.exp(h[batch, truth_n, 0, truth_h, truth_w]) * abs_anchors[truth_n][1]
                )
                predicted_iou = box_iou(full_truth_box, predicted_box)
                tconf[batch, truth_n, :, truth_h, truth_w] = predicted_iou
                conf_learning_scale[batch, truth_n, :, truth_h, truth_w] = 10.0

#        box_learning_scale *= 0.01
        #loss
        x_loss = np.sum((tx - x) ** 2 * box_learning_scale) / 2
        deltas[:,:,0:1,:,:] = (x - tx) * box_learning_scale * (1 - x) * x
        y_loss = np.sum((ty - y) ** 2 * box_learning_scale) / 2
        deltas[:,:,1:2,:,:] = (y - ty) * box_learning_scale * (1 - y) * y
        w_loss = np.sum((tw - w) ** 2 * box_learning_scale) / 2
        deltas[:,:,2:3,:,:] = (w - tw) * box_learning_scale
        h_loss = np.sum((th - h) ** 2 * box_learning_scale) / 2
        deltas[:,:,3:4,:,:] = (h - th) * box_learning_scale
        c_loss = np.sum((tconf - conf) ** 2 * conf_learning_scale) / 2
        deltas[:,:,4:5,:,:] = (conf - tconf) * conf_learning_scale * (1 - conf) * conf
        p_loss = np.sum((tprob - prob) ** 2) / 2
        deltas[:,:,5:,:,:] = ((prob - tprob) * (1 - prob) * prob).transpose(0, 2, 1, 3, 4)
        print("x_loss: %f  y_loss: %f  w_loss: %f  h_loss: %f  c_loss: %f   p_loss: %f" %
            (x_loss, y_loss, w_loss, h_loss, c_loss, p_loss)
        )

        loss = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss

        ret = cls._create_node(loss)
        ret.attrs._output = output
        ret.attrs._deltas = deltas.reshape(batch_size, bbox * (classes + 5), grid_h, grid_w)
        # ret.attrs._cells = cells
        # ret.attrs._bbox = bbox
        # ret.attrs._classes = classes
        return ret

    @classmethod
    def _oper_gpu(cls, output, t, bbox, classes, init_anchors):
        return cls._oper_cpu(output, t, bbox, classes, init_anchors)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._output, Node):
            self.attrs._output._update_diff(context, self.attrs._deltas * dy)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._output, Node):
            self.attrs._output._update_diff(context, get_gpu(self.attrs._deltas) * dy)


class YoloDetector(object):

    def __init__(self, bbox=5, classes=10, init_anchors=None):
        self._bbox = bbox
        self._classes = classes
        self._init_anchors = init_anchors

    def __call__(self, x, y):
        return yolo_detector(x, y, self._bbox, self._classes, self._init_anchors)
