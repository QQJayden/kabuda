# -*- coding:utf-8 -*-

import mxnet as mx
import os
from mxnet.gluon.data.vision import transforms
import pdb

class TrainDataset(mx.gluon.data.Dataset):
    def __init__(self, img_root, annotation_file, img_width, img_height, **kwargs):
        super(TrainDataset, self).__init__(**kwargs)
        with open(annotation_file) as f:
            items_list = [t.split() for t in f.readlines()]
        self.items_dict = {}
        self.items = []
        self.img_width = img_width
        self.img_height = img_height
        self.cur_idx = None
        self.img_root = img_root
        for index, value in enumerate(items_list):
            self.items_dict[value[0] + ' '+  value[1]] = value[2]
            self.items.append(value[0] + ' '+  value[1])
        self.transformer = transforms.Compose([
                transforms.Resize((self.img_width, self.img_height)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __getitem__(self, idx):
        self.cur_idx = idx
        # read in the image H * W * C, image resize and normtransformation to (c,h,w)
        img_path = self.img_root + '/' + self.items[idx]
        img = mx.image.imread(img_path)
        img = self.transformer(img)
        # get the label
        label = 0 if (self.items_dict[self.items[idx]] == 'normal') else 1
        return img, label

    def __len__(self):
        return len(self.items)

if __name__=='__main__':
    data = TrainDataset('train.txt', 2560, 1920)
    pdb.set_trace()
    print('blabla')
