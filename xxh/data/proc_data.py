# -*- coding:utf-8 -*-

import os
import pdb
import random

def travelfolder(rootdir, file_list = []):
    all_files = os.listdir(rootdir)
    for item in all_files:
        item_path = os.path.join(rootdir, item)
        if os.path.isdir(item_path):
            travelfolder(item_path, file_list)
        else:
            file_list.append(item_path)
    return file_list

rootdir = 'train'
train_file = 'train.txt'
val_file = 'val.txt'
image_lists = travelfolder(rootdir)
random.shuffle(image_lists)
with open(train_file, 'w') as train_f:
    with open(val_file, 'w') as val_f:
        for item in image_lists:
            flag_random = random.randrange(20)
            if item.endswith('jpg'):
                if 'zheng_chang' in item:
                    line_write = item + ' ' + 'normal' + '\n'
                else:
                    line_write = item + ' ' + 'flawed' + '\n'

                if flag_random == 19:
                    val_f.write(line_write)
                else:
                    train_f.write(line_write)
