# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MarathonNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # 1: CONV => RELU => POOL
        model.add(Conv2D(32, 5, padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

        # 2: (CONV => RELU) * 2 => POOL
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(1, 1)))

        # 3: (CONV => RELU) * 2 => POOL
        model.add(Conv2D(80, 2, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(80))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
