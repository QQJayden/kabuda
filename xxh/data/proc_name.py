# -*- coding:utf-8 -*-
import os
from xpinyin import Pinyin
import sys
import shutil
import pdb

rootdir = 'train'

list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
pdb.set_trace()
for i in range(0,len(list)):
	path = os.path.join(rootdir,list[i])

	path = path[(len(rootdir)+1):]
	print(path)
	
	pinyin_converter = Pinyin()
        path = unicode(path, 'utf-8')
	res = pinyin_converter.get_pinyin(path, '_')
	res = res.replace(' ', '_')
	res = res.lower()
	print(res)
	src_path = rootdir + '/' + path
	dest_path = rootdir + '/' + res
	print(src_path + '->' + res)
	os.rename(src_path, dest_path)
