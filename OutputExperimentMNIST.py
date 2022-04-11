import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Input

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# multioutput activation experiment
input_layer = Input(shape=input_shape)
first_conv = Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
first_pool = MaxPooling2D(pool_size=(2, 2))(first_conv)
second_conv = Conv2D(64, kernel_size=(3, 3), activation="relu")(first_pool)
second_pool = MaxPooling2D(pool_size=(2, 2))(second_conv)
flatten_layer = Flatten()(second_pool)
dropout_layer = Dropout(0.5)(flatten_layer)
softmax_output = Dense(num_classes, activation="softmax", name='softmax_output')(dropout_layer)
sigmoid_output = Dense(num_classes, activation="sigmoid", name='sigmoid_output')(dropout_layer)

model = Model(inputs=input_layer, outputs=[softmax_output, sigmoid_output])


batch_size = 128
epochs = 25

model.compile(optimizer="adam", loss = {'softmax_output': 'categorical_crossentropy',
					 				    'sigmoid_output': 'categorical_crossentropy'},
					 		    metrics ={'softmax_output': 'accuracy',
					 		    		  'sigmoid_output': 'accuracy'})

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)