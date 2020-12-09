import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,Convolution2D,GlobalMaxPooling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.python.keras.applications.efficientnet import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from PIL import ImageFile
from keras import backend as K
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import hp
import random
import traceback
import pickle
from sklearn.metrics import confusion_matrix, classification_report

ImageFile.LOAD_TRUNCATED_IMAGES = True
train_path = 'C:/Users/Labmint/Documents/Visao/base_padding/training_set'
test_path = 'C:/Users/Labmint/Documents/Visao/base_padding/validation_set'
batch_size = 32

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

#DataGenerator
datagen = ImageDataGenerator(rescale=1., 
    featurewise_center=True,
    rotation_range=10,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="reflect")
train_data = datagen.flow_from_directory( #generator para treino
    train_path,
    target_size=(355, 370),
    batch_size=batch_size,
    class_mode = 'categorical',
    shuffle=False)

testgen = ImageDataGenerator(rescale=1., featurewise_center=True)
test_data = testgen.flow_from_directory( #generator para teste
    test_path,
    target_size=(355, 370),
    batch_size=1,
    class_mode = 'categorical', #use com softmax
    shuffle=False)



#Carregando o modelo
def generate_model(hyper_space):
    model = VGG16(weights='imagenet', include_top=False)
    #print(model.summary())
    x = model.get_layer(hyper_space['last_layer']).output

    #Congelando o treinamento
    model.trainable = False
    for i in range(0,hyper_space['qtd_conv']):
        x = Convolution2D(512, 3,3, activation='relu')(x)

    if hyper_space['pooling']=='AVG':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    else:
        x = GlobalMaxPooling2D(name='max_pool')(x)
    
    for i in range(0, hyper_space['qtd_dense']):
        x =  Dense(hyper_space['neurons'], activation='relu')(x)
    x = Dense(hyper_space['neurons'], activation='relu')(x)
    x = Dropout(hyper_space['dropout'])(x)

    x = Dense(5, activation='softmax')(x)

    for layer in model.layers:
        layer.trainable=False

    model_final = Model(inputs=model.inputs, outputs=x)
    model_final.compile(optimizer=hyper_space['optmizer'], loss="categorical_crossentropy", metrics=["accuracy"])

    return model_final

def train_cnn(hyper_space):
    model = generate_model(hyper_space)
    train_samples = len(train_data.filenames)
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
    model.fit_generator(train_data, epochs = 30, steps_per_epoch=int(train_samples/batch_size),
                        verbose=1,
                        callbacks=[early_stopping])


    y_true = test_data.labels
    y_hat = model.predict_generator(test_data, len(test_data))
    predict_class = np.argmax(y_hat, axis=1)
    predict_class = predict_class.tolist()

    print(classification_report(y_true, predict_class, target_names=['drawings', 'engraving', 'iconography', 'painting', 'sculpture']))
    for layer in model.layers[-15:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit_generator(train_data, epochs=20, steps_per_epoch=int(train_samples/batch_size),
                        verbose=1,
                        callbacks=[early_stopping])
    y_hat = model.predict_generator(test_data, len(test_data))
    predict_class = np.argmax(y_hat, axis=1)
    predict_class = predict_class.tolist()

    acc = metrics.accuracy_score(y_true, predict_class)
    report = classification_report(y_true,predict_class, target_names=['drawings', 'engraving', 'iconography', 'painting', 'sculpture'])
    print(report)

    with open('results.csv','a') as resultado:
        resultado.writerow([acc,hyper_space])

    return model,{
        'loss': 1 - acc,
        'accuracy':acc,
        'report':report,
        'space':hyper_space,
        'status':STATUS_OK
    }


def optmize_cnn():
    try:
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        #max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        train_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=1
    )
    pickle.dump(trials, open("results.pkl", "wb"))
    K.clear_session()


space = {
    'qtd_conv':hp.choice('qrd_conv',[0,1,2,3]),
    'last_layer': hp.choice('last_layer',['block4_conv3','block5_conv3','block3_conv3']),
    'optmizer':hp.choice('optimzer',['adam','Adadelta','Nadam','SGD']),
    'dropout':hp.choice('dropout',[0.1,0.2,0.3]),
    'neurons':hp.choice('neurons',[256,128]),
    'qtd_dense':hp.choice('qtd_dense',[0,1,2,3]),
    'pooling':hp.choice('pooling',['AVG','MAX'])
}

if __name__ == "__main__":
    while True:
        print('Otimizando o modelo')
        try:
            optmize_cnn()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)
