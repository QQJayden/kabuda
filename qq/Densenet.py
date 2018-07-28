
# coding: utf-8

# # 如何修改误差为AUC

# In[1]:

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

train_data = np.load('./data/train.npy')
label_data = np.load('./data/label.npy')
test_data = np.load('./data/test.npy')
filename_data = np.load('./data/filename.npy')

print(train_data.shape,label_data.shape,test_data.shape)


# In[3]:

print(train_data[0].shape)

import cv2

target_train = label_data
X_train = train_data
X_test = test_data

for i in range(2022):
    img_temp_train = cv2.resize(train_data[i],(150,150)).astype(np.int8)
    X_train0.append(img_temp_train)
    
for i in range(662):
    img_temp_test = cv2.resize(test_data[i],(150,150)).astype(np.int8)
    X_test0.append(img_temp_test)
    
X_train0=np.array(X_train0)
X_test0=np.array(X_test0)

print(X_train0.shape,X_test0.shape)
# In[4]:

# 每张图像归一化
target_train = label_data

X_train = []
X_test = []

for img in train_data:
    r = (img[:,:,0]-np.mean(img[:,:,0]))/np.std(img[:,:,0])
    g = (img[:,:,1]-np.mean(img[:,:,1]))/np.std(img[:,:,1])
    b = (img[:,:,2]-np.mean(img[:,:,2]))/np.std(img[:,:,2])
    
    rgb = np.dstack((r,g,b))
    X_train.append(rgb)
    
for img in test_data:
    r = (img[:,:,0]-np.mean(img[:,:,0]))/np.std(img[:,:,0])
    g = (img[:,:,1]-np.mean(img[:,:,1]))/np.std(img[:,:,1])
    b = (img[:,:,2]-np.mean(img[:,:,2]))/np.std(img[:,:,2])
    
    rgb = np.dstack((r,g,b))
    X_test.append(rgb)
    
X_train = np.array(X_train)
X_test=np.array(X_test)
    
print(X_train.shape,X_test.shape)

# Memory error
# for n in X_train:
#     X_train[n,:,:,0] = (X_train[n,:,:,0]-np.mean(X_train[n,:,:,0]))/np.std(X_train[n,:,:,0])
#     X_train[n,:,:,1] = (X_train[n,:,:,1]-np.mean(X_train[n,:,:,1]))/np.std(X_train[n,:,:,1])
#     X_train[n,:,:,2] = (X_train[n,:,:,2]-np.mean(X_train[n,:,:,2]))/np.std(X_train[n,:,:,2])
    
# for n in X_test:
#     X_test[n,:,:,0] = (X_test[n,:,:,0]-np.mean(X_test[n,:,:,0]))/np.std(X_test[n,:,:,0])
#     X_test[n,:,:,1] = (X_test[n,:,:,1]-np.mean(X_test[n,:,:,1]))/np.std(X_test[n,:,:,1])
#     X_test[n,:,:,2] = (X_test[n,:,:,2]-np.mean(X_test[n,:,:,2]))/np.std(X_test[n,:,:,2])


# In[5]:

import keras
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121,DenseNet169
from keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout,Flatten
from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.optimizers import Nadam


def getVggModel():
    
    # VGG16换成其他模型？？
    base_model = DenseNet169(weights='imagenet', include_top=False, 
                 input_shape=X_train.shape[1:], classes=1)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
#     x = GlobalMaxPooling2D()(x)
#     x=Flatten()(x)
#     x = (x-np.mean(x))/(np.max(x)-np.min(x))
    
    # mobile net
#     base_model2 = keras.applications.mobilenet.MobileNet(weights=None, alpha=0.9,input_tensor = base_model.input,include_top=False, input_shape=X_train.shape[1:])
#     base_model2 = keras.applications.mobilenet.MobileNet(weights=None, alpha=0.9,
#                                                          include_top=False, input_shape=X_train.shape[1:])
#     base_model2 = Xception(weights='imagenet', include_top=False, input_tensor = base_model.input,
#                  input_shape=X_train.shape[1:], classes=1)
#     x2 = base_model2.output
#     x2 = GlobalMaxPooling2D()(x2)
        
    merge_one = x
    merge_one = Dense(512, activation='relu', name='fc2')(merge_one)#原来
#     merge_one = Dense(1024, activation='relu', name='fc2')(merge_one)
    merge_one = Dropout(0.3)(merge_one) # 参数原来0.3
    merge_one = Dense(512, activation='relu', name='fc3')(merge_one)
#     merge_one = Dense(256, activation='relu', name='fc3')(merge_one)
    merge_one = Dropout(0.3)(merge_one)
    
    predictions = Dense(1, activation='sigmoid')(merge_one)
    
    model = Model(input=base_model.input, output=predictions)
    
    for layer in base_model.layers:
        layer.trainable = True
    
    # 使用不同的优化
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
    adagrad = Adagrad(lr = 1e-3, epsilon = 1e-6)
    rmsprop = RMSprop(lr=1e-3, rho = 0.9, epsilon=1e-6)
    adadelta = Adadelta(lr=1e-3, rho=0.95, epsilon=1e-06)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    
    # 更换loss
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    return model


# In[6]:

# 自定义损失函数
from sklearn.metrics import roc_auc_score
from keras import backend as K
import tensorflow as tf

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
def jacek_auc(y_true, y_pred):
#     score, up_opt = tf.metrics.auc(y_true, y_pred)
    score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


# In[7]:

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 16 # 原来是3

#Lets define the image transormations that we want
gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         zoom_range=0.2,
                         rotation_range=10,
                         shear_range = 0.05)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_one_input(X1, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=2018)
    while True:
            X1i = genX1.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield X1i[0], X1i[1]

#Finally create out generator
# gen_flow = gen_flow_for_one_inputs(X_train, y_train)

# Finally create generator
def get_callbacks(filepath, patience=2):
   es = EarlyStopping('val_loss', patience=10, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]

'''
epochs_to_wait_for_improve = 10
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
checkpoint_callback = ModelCheckpoint('./model/BestKerasModelResNet50.h5', monitor='val_loss', 
                                      verbose=1, save_best_only=True, mode='min')
'''

#Using K-fold Cross Validation with Data Augmentation.
def mytrainCV(X_train, X_test):
    # K-折交叉验证
    K=3
    
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=2016).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*target_train
    
    auc = 0;
    
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]
        

        #define file path and get callbacks
        file_path = "./model/%s_aug_densenet169_model_weights.hdf5"%j
        callbacks = get_callbacks(filepath=file_path, patience=5)
        gen_flow = gen_flow_for_one_input(X_train_cv, y_train_cv)
        gen_flow_cv = gen_flow_for_one_input(X_holdout, Y_holdout)

        galaxyModel= getVggModel()
    
        # 调整训练参数
        galaxyModel.fit_generator(
                gen_flow,
                steps_per_epoch=len(X_train_cv)//batch_size,
                #steps_per_epoch=100,
                epochs=100,
                shuffle=True,
                verbose=1,
#                 validation_data=gen_flow_cv,
                validation_data=(X_holdout, Y_holdout),
                callbacks=callbacks)

        #Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)
        
        #Getting Training Score
        score = galaxyModel.evaluate(X_train_cv, y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        
        #Getting Test Score
        score = galaxyModel.evaluate(X_holdout, Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Getting validation Score.       
        pred_valid=galaxyModel.predict(X_holdout)
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        #Getting Test Scores
        temp_test=galaxyModel.predict(X_test)
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

        #Getting Train Scores        
        temp_train=galaxyModel.predict(X_train)
        y_train_pred_log+=temp_train.reshape(temp_train.shape[0])
        
        # AUC 
        auc_temp = roc_auc_score(Y_holdout,pred_valid)
        print("AUC = {0:0.4f}".format(auc_temp))
        
        auc+=auc_temp
        
    y_test_pred_log=y_test_pred_log/K
    y_train_pred_log=y_train_pred_log/K
    auc = auc/K

    print('\n Train Loss Validation= ',log_loss(target_train, y_train_pred_log))
    print(' Test Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    print('AUC Validation=',auc)
    return y_test_pred_log


# In[8]:

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss

preds=mytrainCV(X_train,X_test)


# In[ ]:

#Submission for each day.
submission = pd.DataFrame()
submission['filename']=filename_data
submission['probability']=preds
submission.to_csv('./submission/densenet169-1.0.csv',float_format='%.6f',index=False)
# submission.to_csv('./submission/subVgg2.0.csv', 
#                   float_format='%.6f',index=False)

