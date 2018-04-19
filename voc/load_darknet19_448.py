import numpy as np
import renom as rm
from yolov2 import *

classes = 1000
model = Pretrained(classes=classes)
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


        txt= "model.conv%d.beta = rm.Variable(dat[%d:%d].reshape(1, %d, 1, 1))" % (i+1, offset, offset+out_ch, out_ch)
        offset += out_ch
        exec(txt)

        #TODO batch_normalizeのパラメタ
        txt= "model.conv%d.gamma = rm.Variable(dat[%d:%d])" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt= "model.conv%d.bn._mov_mean = rm.Variable(dat[%d:%d])" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt= "model.conv%d.bn._mov_std = rm.Variable(np.sqrt(dat[%d:%d]))" % (i+1, offset, offset+out_ch)
        offset += out_ch
        exec(txt)

        txt= "model.conv%d.conv.params['w'] = rm.Variable(dat[%d:%d].reshape(%d,%d,%d,%d))" % (i+1, offset, offset+(out_ch*in_ch*size*size), out_ch, in_ch, size, size)
        offset += out_ch*in_ch*size*size
        exec(txt)

        print(i, offset)

    in_ch = 1024
    out_ch = classes

    txt= "model.conv23.params['b'] = rm.Variable(dat[%d:%d])" % (offset, offset+out_ch)
    offset += out_ch
    exec(txt)

    txt= "model.conv23.params['w'] = rm.Variable(dat[%d:%d].reshape(%d,%d,1,1))" % (offset, offset+out_ch*in_ch*1, out_ch,in_ch)
    offset += out_ch*in_ch*1
    exec(txt)

    print(dat.shape[0] - offset)

model.save("./weights/darknet19_448.h5")
