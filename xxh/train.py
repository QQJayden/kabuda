# -*- coding: utf-8 -*-

import mxnet
import matplotlib.pyplot as plt
import pdb
import os
import logging
from resnet101 import resnet101_100cls
from mxnet.gluon.data.vision import transforms, datasets
from mxnet import gluon
from mxnet import nd
from time import time
from data.dataset import TrainDataset

# utils functions
def try_gpu():
    try:
        ctx = mxnet.gpu()
        _ = nd.array([0], ctx)
    except:
        ctx = mxnet.cpu()
    return ctx

def evaluate_accuracy(eva_data, net, ctx = mxnet.cpu()):
    acc, n = 0.0, 0
    print("---- evaluating val_accuracy ----")
    for data, label in eva_data:
        n += data.shape[0]
        label = label.astype('float32').as_in_context(ctx)
        output = net(data.as_in_context(ctx))
        right_pred += nd.sum(output.argmax(axis = 1) == label)
    print('Image_num:{}    Acc:{}'.format(n, right_pred / n))
    acc = right_pred / n
    return acc

# configs
train_batch_size = 1
val_batch_size = 32
momentum = 0.9
lr = 1e-3
lr_decay = 0.1
lr_period = 5
wd = 1e-5
epochs = 50
ctx = try_gpu()

# make train_data_loader and test_data_loader
pdb.set_trace()
train_data = TrainDataset(img_root = 'data', annotation_file = 'data/train.txt', img_width = 800, img_height = 600)
val_data = TrainDataset(img_root = 'data', annotation_file = 'data/val.txt', img_width = 800, img_height = 600)

train_loader = gluon.data.DataLoader(train_data, batch_size = train_batch_size, shuffle = True, last_batch = 'keep')
val_loader = gluon.data.DataLoader(val_data, batch_size = val_batch_size, shuffle = True, last_batch = 'keep')

# build the net and initialize it
resnet = resnet101_100cls()
resnet.output.initialize(ctx = ctx)
resnet.hybridize()
soft_loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
trainer = mxnet.gluon.Trainer(resnet.collect_params(),'sgd',
                              {'learning_rate':lr, 'momentum':0.9, 'wd': wd })
logging.basicConfig(filename = 'train.log', level = logging.INFO)
logging.info('Traing starts.')

# start train the net
for epoch in range(epochs):
    # adjust the learning_rate
    if epoch > 0 and epoch % lr_period == 0:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)
    train_loss, train_acc, n  = 0.0, 0.0, 0
    for data, label in train_loader:
        #pdb.set_trace()
        n += 1
        label = label.astype('float32').as_in_context(ctx)
        with mxnet.autograd.record():
            output = resnet(data.as_in_context(ctx))
            loss = soft_loss(output, label)
        loss.backward()
        trainer.step(batch_size = train_batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += nd.mean(output.argmax(axis = 1) == label).asscalar()
        logging.info('Epoch {}. Iter {}. Train_loss {:.3f} Train_acc {:.3f}'\
              .format(epoch, n, train_loss / n, train_acc / n))
        print('Epoch {}. Iter {}. Train_loss {:.3f} Train_acc {:.3f}'\
              .format(epoch, n, train_loss / n, train_acc / n))

    val_acc = evaluate_accuracy(val_loader, resnet, ctx)
    logging.info('Epoch:{}  Train_loss:{:.3f}  Train_acc:{:.3f} Val_acc:{:.3f}'\
          .format(epoch, train_loss / n, train_acc / n, val_acc))
    print('Epoch:{}  Train_loss:{:.3f}  Train_acc:{:.3f} Val_acc:{:3f}'\
          .format(epoch, train_loss / n, train_acc / n, val_acc))
    resnet.export('output/resnet', epoch)
