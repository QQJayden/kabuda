
# coding: utf-8

# ### 直接从文件中读取

# In[ ]:

import os
from scipy.misc import imread
import numpy as np
import shutil

get_ipython().run_line_magic('matplotlib', 'inline')

detect_path = './data/train2/train/detect'
normal_path = './data/train2/train/normal'

# 数据分类存入train2
# part1
g = os.walk("./data/part1")
print('part1')
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):
            if path.endswith('正常'):
                shutil.copy(os.path.join(path, filename),
                            normal_path)
            else:
                shutil.copy(os.path.join(path, filename),
                            detect_path)

            print (os.path.join(path, filename))
            
# part1
g = os.walk("./data/part2")
print('part2')
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):
            if path.endswith('正常'):
                shutil.copy(os.path.join(path, filename),
                            normal_path)
            else:
                shutil.copy(os.path.join(path, filename),
                            detect_path)

            print (os.path.join(path, filename))
            
            
# part1
g = os.walk("./data/part3")
print('part3')
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):
            if path.endswith('正常'):
                shutil.copy(os.path.join(path, filename),
                            normal_path)
            else:
                shutil.copy(os.path.join(path, filename),
                            detect_path)

            print (os.path.join(path, filename))


# In[3]:

from keras.applications.densenet import DenseNet121,DenseNet169
from keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout,Flatten
from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.preprocessing import image

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.optimizers import Nadam

base_model = DenseNet121(weights='imagenet', include_top=False, 
                         input_shape=(224,224,3), classes=1)
x = base_model.output
x = GlobalAveragePooling2D()(x)
    
merge_one = x
merge_one = Dense(512, activation='relu', name='fc2')(merge_one)#原来
merge_one = Dropout(0.3)(merge_one) # 参数原来0.3
merge_one = Dense(512, activation='relu', name='fc3')(merge_one)
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


# In[10]:

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_data_dir = './data/train2/train'
validation_data_dir = './data/train2/validation'

img_height = 224
img_width = 224

# Finally create generator
def get_callbacks(filepath, patience=2):
   es = EarlyStopping('val_loss', patience=10, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]

model_path = "./model/%s_aug_DesNet_model_weights.hdf5"%j
callbacks = get_callbacks(filepath=model_path, patience=5)

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='binary')


# fine-tune the model
model.fit_generator(
    train_generator,
#     samples_per_epoch=2022,
    steps_per_epoch = 2022//batch_size,
    epochs=100,
    shuffle = True,
    verbose =1,
    validation_data=validation_generator,
#     nb_val_samples=200,
    callbacks = callbacks)


# In[ ]:

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

test_model = load_model('my_model_name.h5')
img = load_img('image_to_predict.jpg',False,target_size=(img_width,img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
print(preds, probs)


# In[ ]:

galaxyModel = model
galaxyModel.load_weights(filepath=model_path)



# #Getting Training Score
# score = galaxyModel.evaluate(X_train_cv, y_train_cv, verbose=0)
# print('Train loss:', score[0])
# print('Train accuracy:', score[1])

# #Getting Test Score
# score = galaxyModel.evaluate(X_holdout, Y_holdout, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# #Getting validation Score.       
# pred_valid=galaxyModel.predict(X_holdout)
# y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

#Getting Test Scores
temp_test=galaxyModel.predict(X_test)
y_test_pred_log=temp_test.reshape(temp_test.shape[0])

# #Getting Train Scores        
# temp_train=galaxyModel.predict(X_train)
# y_train_pred_log+=temp_train.reshape(temp_train.shape[0])
        
# AUC 
auc_temp = roc_auc_score(Y_holdout,pred_valid)
print("AUC = {0:0.4f}".format(auc_temp))

pred = y_test_pred_log


# In[ ]:

#Submission for each day.
submission = pd.DataFrame()
submission['filename']=filename_data
submission['probability']=preds
submission.to_csv('./submission/densenet169-1.0.csv',float_format='%.6f',index=False)
# submission.to_csv('./submission/subVgg2.0.csv', 
#                   float_format='%.6f',index=False)

