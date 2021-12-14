import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input


def make_model(learning_rate=0.0001, size_inner=256, drop_rate=0.2):
    pre_trained_model = Xception(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    pre_trained_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    base = pre_trained_model(inputs, training=False)
    x = keras.layers.MaxPooling2D(2, 2)(base)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(size_inner, activation='relu')(x)
    x = keras.layers.Dropout(drop_rate)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, x)
    
    model.compile(optimizer = RMSprop(lr=learning_rate),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
    
    return model


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_model.h5', #the name of the HDF5 model that is saved with callbacks
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


train_datagen_tl = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen_tl.flow_from_directory(
        'Tire Textures/training_data',   #dir of the training_data
        target_size=(224, 224), 
        batch_size=20,
        class_mode='binary',
        shuffle=True)


validation_datagen_tl = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen_tl.flow_from_directory(
        'Tire Textures/testing_data',  #dir of the testing_data
        target_size=(224, 224), 
        batch_size=20,
        class_mode='binary',
        shuffle=True)


model = make_model()
history = model.fit(train_generator,epochs=10,validation_data=validation_generator,callbacks=[checkpoint])

best_model = keras.models.load_model('xception_model.h5') # you can change the name of HDF5 file by the name of the HDF5 file that is saved by callbacks
best_model.save('xception_model') # save the model so that it can be served through tensorflow serving