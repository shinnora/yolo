import numpy as np
import cv2
import os
import sys
from xml.etree import ElementTree
from itertools import product
import urllib.request as request


class VOCgenerator():

    def __init__(self, ann_path="VOCdevkit/VOC2012/Annotations/", img_path="VOCdevkit/VOC2012/JPEGImages/"):
        self.img_path = img_path
        self.train_file_list = [path for path in sorted(os.listdir(ann_path)) if not "2012_" in path]
        self.test_file_list = [path for path in os.listdir(ann_path) if "2012_" in path]
        self.tree = ElementTree.parse(os.path.join(dataset_path, train_file_list[-1]))
        self.cat_num = 20
        self.label_dict = {}

    def one_hot(self, id):
        one_hot_label = np.zeros(self.cat_num)
        one_hot_label[id] = 1
        return one_hot_label

    def generate_samples(self, batch_size, size=None, train=True):
        if train:
            file_list = self.train_file_list
        else:
            file_list = self.test_file_list
        img_order = np.random.permutation(len(file_list))
        for batch_num in range(int(len(img_order) / batch_size)):
            x = []
            t = []
            for img_num in img_order[batch_num * batch_size : (batch_num + 1) * batch_size]:
                file_name, ground_truths = self._get_img_info(file_list[img_num])
                img_path = os.path.join(self.img_path, file_name)
                if size is None:
                    img = reshape_to_yolo_size(cv2.imread(img_path))
                else:
                    img = cv2.resize(cv2.imread(img_path), size)
                img = np.asarray(img, dtype=np.float32) / 255.0
                img = img.transpose(2, 0, 1)
                x.append(img)
                t.append(ground_truths)
            yield x, t

    def _get_obj_coordinate(self, obj):
        class_name = obj.find("name").text.strip()
        if self.label_dict.get(class_name, None) is None:
            self.label_dict[class_name] = len(self.label_dict)
        class_id = label_dict[class_name]
        bbox = obj.find("bndbox")
        xmax = float(bbox.find("xmax").text.strip())
        xmin = float(bbox.find("xmin").text.strip())
        ymax = float(bbox.find("ymax").text.strip())
        ymin = float(bbox.find("ymin").text.strip())
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w/2
        y = ymin + h/2
        return class_id, x, y, w, h

    def _get_img_info(self, filename):
        tree = ElementTree.parse(filename)
        node = tree.getroot()
        file_name = node.find("filename").text.strip()
        img_h = float(node.find("size").find("height").text.strip())
        img_w = float(node.find("size").find("width").text.strip())
        obj_list = node.findall("object")
        ground_truths = []
        for obj in obj_list:
            class_id, x, y, w, h = self._get_obj_coordinate(obj)
            ground_truths.append({
                "x" : x / img_w,
                "y" : y / img_h,
                "w" : w / img_w,
                "h" : h / img_h,
                "label" : class_id,
                "one_hot_label" : self.one_hot(class_id)
            })
        return file_name, ground_truths



def reshape_to_yolo_size(img):
    input_height, input_width, _ = img.shape
    min_pixel = 320
    #max_pixel = 608
    max_pixel = 448

    min_edge = np.minimum(input_width, input_height)
    if min_edge < min_pixel:
        input_width *= min_pixel / min_edge
        input_height *= min_pixel / min_edge
    max_edge = np.maximum(input_width, input_height)
    if max_edge > max_pixel:
        input_width *= max_pixel / max_edge
        input_height *= max_pixel / max_edge

    input_width = int(input_width / 32 + round(input_width % 32 / 32)) * 32
    input_height = int(input_height / 32 + round(input_height % 32 / 32)) * 32
    img = cv2.resize(img, (input_width, input_height))

    return img
