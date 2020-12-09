from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.preprocessing import image_dataset_from_directory
from keras.models import load_model, Model
from tensorflow.keras.applications import EfficientNetB6, MobileNetV2, VGG16
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.optimizers import SGD
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import tensorflow as tf

#Path do dataset
batch = 1
train_path='C:/Users/Labmint/Documents/Visao/base_padding/training_set'
test_path='C:/Users/Labmint/Documents/Visao/base_padding/validation_set'



train_set = image_dataset_from_directory(train_path,batch_size=batch, labels='inferred',label_mode='int',
                                         image_size=(355,370),
                                         shuffle = False
                                         )
test_set = image_dataset_from_directory(test_path,batch_size=batch, labels='inferred',label_mode='int',
                                         image_size=(355,370),
                                         shuffle = False
                                         )



#Carregando a rede
#As outras redes podem ser importadas VGG16, MobileNetV2
model = EfficientNetB6(include_top=False,input_shape=(355,370,3), pooling = 'avg', weights='imagenet')

train_feats=[]
y_train = []

for img in train_set:
  img_train= img[0].numpy()
  #img_train = tf.keras.applications.mobilenet_v2.preprocess_input(img_train)
  #img_train = tf.keras.applications.vgg16.preprocess_input(img_train)
  predicts = model.predict(img_train)
  y_train.append(img[1].numpy()[0])
  train_feats.append(predicts.squeeze())

test_feats=[]
y_test = []

for img in test_set:
  img_test = img[0].numpy()
  predicts = model.predict(img_test)
  y_test.append(img[1].numpy()[0])
  test_feats.append(predicts.squeeze())

#Carregando o classificador
svm_model = SVC(kernel='poly', C=1, decision_function_shape='ovo', degree=3,verbose=True)
x_train_feats = train_feats
x_train_feats = np.array(x_train_feats)

x_test_feats = test_feats
x_test_feats = np.array(x_test_feats)

#Treinando o classificador
svm_model.fit(x_train_feats, y_train)

#Predict
y_hat = svm_model.predict(x_test_feats)
test_acc = metrics.accuracy_score(y_test, y_hat)
print("Test Accuracy score {0}".format(test_acc))

