import numpy as np
import renom as rm
from renom.cuda.cuda import set_cuda_active, cuGetDeviceCount
from renom import cuda
from renom.utility.trainer import Trainer
from renom.utility.distributor import NdarrayDistributor
from darknet19 import *

set_cuda_active(True)

# hyper parameters
img_size = (224, 224)
background_path = "./backgrounds"
# label_file = "./data/label.txt"
backup_path = "./backup"
batch_size = 32
max_epochs = 3000
learning_rate = 0.01
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005
classes = 10
num_gpu = cuGetDeviceCount()

# load image generator
print("loading image generator...")

# with open(label_file, "r") as f:
#     labels = f.read().strip().split("\n")

# load model
print("loading model...")
model = Darknet19(classes)
backup_file = "%s/backup.h5" % (backup_path)
# if os.path.isfile(backup_file):
#     model.load(backup_file)
#cuda.get_device(0).use()
#model.to_gpu() # for gpu

trainer = Trainer(model,
                  batch_size=batch_size,
                  loss_func=rm.mean_squared_error,
                  num_epoch=1,
                  optimizer=rm.Sgd(lr=learning_rate, momentum=momentum), num_gpu=num_gpu)


# start to train
print("start training")
for epoch in range(max_epochs):
    generator = VOCgenerator().generate_samples(batch_size=batch_size, size=img_size)
    while True:
        # generate sample
        try:
            x, t = next(generator)
        except StopIteration:
            break
        #x = rm.Variable(x)
        one_hot_t = []
        for i in range(len(t)):
            one_hot_t.append(t[i][0]["one_hot_label"])
        #x.to_gpu()
        one_hot_t = np.array(one_hot_t, dtype=np.float32)
        #one_hot_t = rm.Variable(one_hot_t)
        #one_hot_t.to_gpu()
        trainer.train(train_distributor=NdarrayDistributor(x, one_hot_t))
        # with model.train():
        #     output = model(x)
        #     loss = rm.softmax_cross_entropy(output, one_hot_t)

        #loss.to_cpu()

        # grad = loss.grad()
        # grad.update(opt)
        # print("[batch %d (%d images)] loss: %f" % (batch+1, (batch+1) * batch_size, loss))

        trainer.optimizer = rm.Sgd(lr=learning_rate * (1 - batch / max_batches) ** lr_decay_power, momentum=momentum) # Polynomial decay learning rate

    # save model
    if (epoch+1) % 100 == 0:
        model_file = "%s/%s.h5" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        model.save(model_file)
        model.save(backup_file)
