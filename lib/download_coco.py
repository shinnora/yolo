from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import cv2
from lib.utils import *
import matplotlib.pyplot as plt

class CocoGenerator():

    def __init__(self, dataDir='../data/', dataType='val2017'):
        self.annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
        self.coco = COCO(self.annFile)
        self.catNum = len(self.coco.getCatIds())
        self.imgIds = self.coco.getImgIds()

    def one_hot(self, id):
        one_hot_label = np.zeros(self.catNum)
        one_hot_label[id] = 1
        return one_hot_label

    def generate_samples(self, batch_size, size=None):
        imgorder = np.random.permutation(self.imgIds)
        for batch_num in range(int(len(imgorder) / batch_size)):
            x = []
            t = []
            for imgId in imgorder[batch_num * batch_size : (batch_num + 1) * batch_size]:
                ground_truths = []
                img = self.coco.loadImgs(ids=[imgId])[0]
                if size is None:
                    Img = reshape_to_yolo_size(io.imread(img['coco_url']))
                else:
                    Img = cv2.resize(io.imread(img['coco_url']), size)
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[imgId]))
                for ann in anns:
                    bbox = ann['bbox']
                    cat_id = ann['category_id']
                    ground_truths.append({
                        "x": bbox[0] / img['width'],
                        "y": bbox[1] / img['height'],
                        "w": bbox[2] / img['width'],
                        "h": bbox[3] / img['height'],
                        "label": cat_id-1,
                        "one_hot_label": self.one_hot(cat_id-1)
                    })
                Img = np.asarray(Img, dtype=np.float32) / 255.0
                Img = Img.transpose(2, 0, 1)
                x.append(Img)
                t.append(ground_truths)
            yield x, t

            

class ImageNetGenerator():

    def __init__
