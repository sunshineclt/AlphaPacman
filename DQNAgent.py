from keras.initializers import random_normal
from keras.layers import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, learning_rate, img_rows, img_cols, img_channels):
        self.model = self.build_model(learning_rate, img_rows, img_cols, img_channels)

    def build_model(self, learning_rate, img_rows, img_cols, img_channels):
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding="same",
                         input_shape=(img_rows, img_cols, img_channels),
                         kernel_initializer=random_normal(stddev=0.01)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                         kernel_initializer=random_normal(stddev=0.01)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                         kernel_initializer=random_normal(stddev=0.01)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=random_normal(stddev=0.01)))
        model.add(Activation('relu'))
        model.add(Dense(9, kernel_initializer=random_normal(stddev=0.01)))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        print("We finish building the model")
        return model
