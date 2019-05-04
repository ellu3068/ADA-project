import time
from tensorflow import keras
import math
from tensorflow.keras.callbacks import ModelCheckpoint


# Evaluation Time callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# MODEL 1
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model1/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model1_time = sum(time_callback.times)



## MODEL 1 WITH DROPOUT LAYER TO COMBAT OVERFITTING
# rate = 0.25

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model1/Dropout/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/Dropout/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model1_drop_time = sum(time_callback.times)


## MODEL 1 WITH BATCHNORMALIZATION

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model1/Norm/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/Norm/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 25,validation_data=validation_generator, callbacks=callbacks_list)

model1_norm_time = sum(time_callback.times)



## MODEL 1 WITH DROPOUT + BATCHNORMALIZATION
# rate = 0.25

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 10, activation = 'softmax'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
model_dir ='./CIFAR-10_Models/Model1'
tbCallBack = keras.callbacks.TensorBoard(log_dir= model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath = model_dir+"/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

print('MODEL 1 WITH DROPOUT + BATCH NORMALIZATION')

model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs = 50, validation_data=validation_generator, callbacks=callbacks_list,verbose = 2)

times = time_callback.times


#---------------------------------------------------------------

import time
from tensorflow import keras
import math
from tensorflow.keras.callbacks import ModelCheckpoint


# Evaluation Time callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def create_my_inception_layer(previous_layer, output_dimension, layer_name):
#Check if using a single bottleneck layer will reduce dimension?

    dimension = math.floor(int(output_dimension/5))
    bottleneck_dim = math.floor(dimension/2)

    tower_1 = keras.layers.Conv2D(filters=dimension, kernel_size=(1, 1), activation='relu', padding = 'same')(previous_layer)

    bottleneck2 = keras.layers.Conv2D(filters=bottleneck_dim, kernel_size=(1, 1), activation='relu', padding = 'same')(previous_layer)
    tower_2 = keras.layers.Conv2D(filters=dimension, kernel_size=(2, 2), activation='relu', padding = 'same')(bottleneck2)

    bottleneck3 = keras.layers.Conv2D(filters=bottleneck_dim, kernel_size=(1, 1), activation='relu', padding = 'same')(previous_layer)
    tower_3 = keras.layers.Conv2D(filters=dimension, kernel_size=(3, 3), activation='relu', padding = 'same')(bottleneck3)

    bottleneck4 = keras.layers.Conv2D(filters=bottleneck_dim, kernel_size=(1, 1), activation='relu', padding = 'same')(previous_layer)
    tower_4 = keras.layers.Conv2D(filters=dimension, kernel_size=(5, 5), activation='relu', padding = 'same')(bottleneck4)

    bottleneck5 = keras.layers.Conv2D(filters=bottleneck_dim, kernel_size=(1, 1), activation='relu', padding = 'same')(previous_layer)
    tower_5 = keras.layers.Conv2D(filters=dimension, kernel_size=(7, 7), activation='relu', padding = 'same')(bottleneck5)

    inception_layer = keras.layers.concatenate([tower_1,tower_2,tower_3, tower_4, tower_5 ], axis = 3, name = layer_name)

    return inception_layer


### Model3

input_tensor = keras.Input(shape = (64,64,3))

inception_layer_1 = create_my_inception_layer(input_tensor, 64 , 'inception1')

pooling_layer1 = keras.layers.MaxPooling2D(pool_size=(2,2))(inception_layer_1)

inception_layer_2 = create_my_inception_layer(pooling_layer1, 128 , 'inception2')

pooling_layer2 = keras.layers.MaxPooling2D(pool_size=(2,2))(inception_layer_2)

inception_layer_3 = create_my_inception_layer(pooling_layer2, 256 , 'inception3')

pooling_layer3 = keras.layers.MaxPooling2D(pool_size=(2,2))(inception_layer_3)

flatten_layer = keras.layers.Flatten()(pooling_layer3)

FC1 = keras.layers.Dense(units = 512, activation = 'relu')(flatten_layer)

FC2 = keras.layers.Dense(units = 128, activation = 'relu')(FC1)

output_layer = keras.layers.Dense(units = 1, activation = 'sigmoid')(FC2)

model = keras.models.Model(input_tensor, output_layer)

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
model_dir ='./Model3'
tbCallBack = keras.callbacks.TensorBoard(log_dir= model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath = model_dir+"./checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

times = time_callback.times


### MODEL 3 WITH DROPOUT

input_tensor = keras.Input(shape = (64,64,3))

inception_layer_1 = create_my_inception_layer(input_tensor, 64 , 'inception1')

dropout_1 = keras.layers.Dropout(rate = 0.25)(inception_layer_1)

pooling_layer1 = keras.layers.MaxPooling2D(pool_size=(2,2))(dropout_1)

inception_layer_2 = create_my_inception_layer(pooling_layer1, 128 , 'inception2')

dropout_2 = keras.layers.Dropout(rate = 0.25)(inception_layer_2)

pooling_layer2 = keras.layers.MaxPooling2D(pool_size=(2,2))(dropout_2)

inception_layer_3 = create_my_inception_layer(pooling_layer2, 256 , 'inception3')

dropout_3 = keras.layers.Dropout(rate = 0.25)(inception_layer_3)

pooling_layer3 = keras.layers.MaxPooling2D(pool_size=(2,2))(dropout_3)

flatten_layer = keras.layers.Flatten()(pooling_layer3)

FC1 = keras.layers.Dense(units = 512, activation = 'relu')(flatten_layer)

dropout_4 = keras.layers.Dropout(rate = 0.25)(FC1)

FC2 = keras.layers.Dense(units = 128, activation = 'relu')(dropout_4)

dropout_5 = keras.layers.Dropout(rate = 0.25)(FC2)

output_layer = keras.layers.Dense(units = 1, activation = 'sigmoid')(dropout_5)

model = keras.models.Model(input_tensor, output_layer)

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
model_dir ='./Model3/Drop'
tbCallBack = keras.callbacks.TensorBoard(log_dir= model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath = model_dir+"/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 26,validation_data=validation_generator, callbacks=callbacks_list,verbose = 2)

times = time_callback.times


### MODEL 3 WITH BATCHNORMALIZATION

input_tensor = keras.Input(shape = (64,64,3))

inception_layer_1 = create_my_inception_layer(input_tensor, 64 , 'inception1')

BN_1 = keras.layers.BatchNormalization()(inception_layer_1)

pooling_layer1 = keras.layers.MaxPooling2D(pool_size=(2,2))(BN_1)

inception_layer_2 = create_my_inception_layer(pooling_layer1, 128 , 'inception2')

BN_2 = keras.layers.BatchNormalization()(inception_layer_2)

pooling_layer2 = keras.layers.MaxPooling2D(pool_size=(2,2))(BN_2)

inception_layer_3 = create_my_inception_layer(pooling_layer2, 256 , 'inception3')

BN_3 = keras.layers.BatchNormalization()(inception_layer_3)

pooling_layer3 = keras.layers.MaxPooling2D(pool_size=(2,2))(BN_3)

flatten_layer = keras.layers.Flatten()(pooling_layer3)

FC1 = keras.layers.Dense(units = 512, activation = 'relu')(flatten_layer)

FC2 = keras.layers.Dense(units = 128, activation = 'relu')(FC1)

output_layer = keras.layers.Dense(units = 1, activation = 'sigmoid')(FC2)

model = keras.models.Model(input_tensor, output_layer)

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
model_dir ='./Model3/Norm'
tbCallBack = keras.callbacks.TensorBoard(log_dir= model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath = model_dir+"/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

print('MODEL 3 WITH BATCH NORMALIZATION')

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 25,validation_data=validation_generator, callbacks=callbacks_list,verbose = 2)

times = time_callback.times




# MODEL 3 WITH DROPOUT + BATCHNORMALIZATION

input_tensor = keras.Input(shape = (64,64,3))

inception_layer_1 = create_my_inception_layer(input_tensor, 64 , 'inception1')

BN_1 = keras.layers.BatchNormalization()(inception_layer_1)

pooling_layer1 = keras.layers.MaxPooling2D(pool_size=(2,2))(BN_1)

inception_layer_2 = create_my_inception_layer(pooling_layer1, 128 , 'inception2')

BN_2 = keras.layers.BatchNormalization()(inception_layer_2)

pooling_layer2 = keras.layers.MaxPooling2D(pool_size=(2,2))(BN_2)

inception_layer_3 = create_my_inception_layer(pooling_layer2, 256 , 'inception3')

BN_3 = keras.layers.BatchNormalization()(inception_layer_3)

pooling_layer3 = keras.layers.MaxPooling2D(pool_size=(2,2))(BN_3)

flatten_layer = keras.layers.Flatten()(pooling_layer3)

FC1 = keras.layers.Dense(units = 512, activation = 'relu')(flatten_layer)

dropout_4 = keras.layers.Dropout(rate = 0.25)(FC1)

FC2 = keras.layers.Dense(units = 128, activation = 'relu')(dropout_4)

dropout_5 = keras.layers.Dropout(rate = 0.25)(FC2)

output_layer = keras.layers.Dense(units = 10, activation = 'softmax')(dropout_5)

model = keras.models.Model(input_tensor, output_layer)

# load weights
model.load_weights(filepath)

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
### Using tensorboard callbacks
model_dir ='./CIFAR-10_Models/Model3/DropNorm2'
tbCallBack = keras.callbacks.TensorBoard(log_dir= model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath = model_dir+"/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

print('MODEL 3 WITH DROPOUT + BATCH NORMALIZATION')

model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs = 50, validation_data=validation_generator, callbacks=callbacks_list,verbose = 2)

times = time_callback.times

from sklearn.metrics import classification_report
import numpy as np

x_test = image_list2

y_test = np.ones(1000) * 9
_
y_pred = model.predict(x_test)

y_classes = np.argmax(y_pred, axis=-1)

print(classification_report(y_test, y_classes))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_classes))

model.evaluate(validation_generator)

from PIL import Image
import glob

image_list = []
for filename in glob.glob('cifar-10/cifar-10_train_test/Testing/truck/*.png'):
    im=Image.open(filename)
    im = im.resize((64,64))
    im = np.array(im)
    image_list.append(im)

image_list2 = np.asarray(image_list)
