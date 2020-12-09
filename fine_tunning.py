from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing import image_dataset_from_directory
from keras.models import load_model, Model
from tensorflow.keras.applications import EfficientNetB6, VGG16, MobileNetV2
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn import metrics
import numpy as np

train_path = 'C:/Users/Labmint/Documents/Visao/base_padding/training_set'
test_path = 'C:/Users/Labmint/Documents/Visao/base_padding/validation_set'
batch_size = 64

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
    shuffle=True)

testgen = ImageDataGenerator(rescale=1., featurewise_center=True)
test_data = testgen.flow_from_directory( #generator para teste
    test_path,
    target_size=(355, 370),
    batch_size=1,
    class_mode = 'categorical', #use com softmax
    shuffle=False)

#Carregando o modelo
#As outras redes podem ser importadas EfficientNetB6, MobileNetV2
model = VGG16(weights='imagenet', include_top=False)
#Congelando o treinamento
model.trainable = False

#Adicionamento as camadas
x = GlobalAveragePooling2D(name='avg_pool')(model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(5, activation='softmax')(x)

model = Model(inputs=model.inputs, outputs=x)

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
train_samples = len(train_data.filenames)
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
model.fit_generator(train_data, epochs = 30, steps_per_epoch=int(train_samples/batch_size),
                    verbose=1,
                    callbacks=[early_stopping])

from sklearn.metrics import confusion_matrix, classification_report

#Métricas com a rede congelada
y_true = test_data.labels
y_hat = model.predict_generator(test_data, len(test_data))
predict_class = np.argmax(y_hat, axis=1)
predict_class = predict_class.tolist()

print(classification_report(y_true, predict_class, target_names=['drawings', 'engraving', 'iconography', 'painting', 'sculpture']))

#Descongelando as camadas
#A quantidade de camadas muda de modelo para modelo
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

#Métrica com a rede descongelada
y_hat = model.predict_generator(test_data, len(test_data))
predict_class = np.argmax(y_hat, axis=1)
predict_class = predict_class.tolist()

print(classification_report(y_true, predict_class, target_names=['drawings', 'engraving', 'iconography', 'painting', 'sculpture']))