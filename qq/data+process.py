
# coding: utf-8

# In[ ]:

# 处理xml文件

import xml.etree.cElementTree as et
'''
调用xml库中的相关方法可以快速实现xml文件中的信息的读取
此处读取的xml文件的格式为VOC中的数据标注格式，以下脚本展示了
读取所有的bounding box的坐标的方法，相应的可以按照以下方式
读取其他的信息。
'''

img_path = './data/part1/吊纬/'

tree=et.parse(img_path + 'J01_2018.06.13 13_25_43.xml')
root=tree.getroot()

filename=root.find('filename').text
print(filename)

for Object in root.findall('object'):
    name=Object.find('name').text
    print(name)
    bndbox=Object.find('bndbox')
    xmin=bndbox.find('xmin').text
    ymin=bndbox.find('ymin').text
    xmax=bndbox.find('xmax').text
    ymax=bndbox.find('ymax').text
    print(xmin,ymin,xmax,ymax)

# from PIL import Image
# import matplotlib.pyplot as plt
# img = Image.open('./data/part1/吊纬/J01_2018.06.17 09_09_56.jpg')
# # img = img.resize((224,224))
# plt.figure("img")
# plt.imshow(img)
# plt.show()

import cv2
img = cv2.imread('./data/part1/吊纬/J01_2018.06.13 13_25_43.jpg')

cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255))
cv2.imshow('img',img)
cv2.waitKey(0)

%matplotlib inline

# import cv2
# img = cv2.imread('./data/part1/吊纬/J01_2018.06.13 13_25_43.jpg')
# cv2.imshow('img',img)

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('./data/part1/吊纬/J01_2018.06.17 09_09_56.jpg')
img = img.resize((224,224))
plt.figure("img")
plt.imshow(img)
plt.show()

print(img.size)import os

for filename in os.listdir(r"./data/part1/正常"):              #listdir的参数是文件夹的路径
    print ( filename)                                  #此时的filename是文件夹中文件的名称
    
# In[1]:

import os
from scipy.misc import imread
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train_part1 = []
train_part2 = []
train_part3 = []

train = []

label = []


# 读取三个part下所有图片
# part1
g = os.walk("./data/part1")
print('part1')
for path,d,filelist in g:  
    
    for filename in filelist:
        
        
        if filename.endswith('jpg'):
            if path.endswith('正常'):
                label_temp = 0
            else:
                label_temp = 1
            
            label.append(label_temp)
            
            img_temp = imread(os.path.join(path, filename))
            img_temp = cv2.resize(img_temp,(360,360)).astype(np.float32)                
            train.append(img_temp)
            print (os.path.join(path, filename),img_temp.shape,label_temp)

# plt.imshow(img_temp)        

# part2
g = os.walk("./data/part2")
print('part2')
for path,d,filelist in g:  
    for filename in filelist:
        if filename.endswith('jpg'):
            if path.endswith('正常'):
                label_temp = 0
            else:
                label_temp = 1
            
            label.append(label_temp)
            
            img_temp = imread(os.path.join(path, filename))
            img_temp = cv2.resize(img_temp,(360,360)).astype(np.float32)
            train.append(img_temp)
            print (os.path.join(path, filename),img_temp.shape,label_temp)
            
            
g = os.walk("./data/part3")
print('part3')
for path,d,filelist in g:  
    for filename in filelist:

        if filename.endswith('jpg'):
            if path.endswith('正常'):
                label_temp = 0
            else:
                label_temp = 1
            
            label.append(label_temp)
            img_temp = imread(os.path.join(path, filename))
            img_temp = cv2.resize(img_temp,(360,360)).astype(np.float32)
            train.append(img_temp)
            print (os.path.join(path, filename),img_temp.shape,label_temp)

#             for i in range(1,4):
#                 img_temp[:,:,i].resize(224,224) 
#             plt.imshow(img_temp)

    


# In[2]:

# 测试数据集
g = os.walk("./data/test")
test = []
name = []
print('test')
for path,d,filelist in g:  
    for filename in filelist:
        if filename.endswith('jpg'):
            img_temp = imread(os.path.join(path, filename))
            img_temp = cv2.resize(img_temp,(360,360)).astype(np.float32)                
            test.append(img_temp)
            name.append(filename)
            print (os.path.join(path, filename),img_temp.shape)


# In[3]:

plt.imshow(img_temp/255)

train_part1 = np.array(train_part1)
train_part2 = np.array(train_part2)
train_part3 = np.array(train_part3)
print(train_part1.shape,train_part2.shape,train_part3.shape)
# In[3]:

train = np.array(train)
label = np.array(label)
test = np.array(test)
name = np.array(name)

print(train.shape,label.shape,test.shape,name.shape)
# print(train[0,:,:,0])
# print(label[0])


# In[ ]:

np.savez("data_all.npz",train = train,label = label, test = test, name = name)


# In[4]:

# 保存训练数据
np.save('./data/train360.npy',train)
np.save('./data/label360.npy',label)
np.save('./data/test360.npy',test)
np.save('./data/filename360.npy',name)

print('done!')


# In[2]:

import numpy as np

train_data = np.load('./data/train360.npy')
label_data = np.load('./data/label360.npy')
test_data = np.load('./data/test360.npy')
filename_data = np.load('./data/filename360.npy')

print(train_data.shape,label_data.shape,test_data.shape,filename_data.shape)


# In[ ]:

np.savez_compressed("./data/data_all.npz",train = train_data,label = label_data, test = test_data, name = filename_data)

