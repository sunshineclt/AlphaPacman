from keras.layers import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

IMG_ROWS, IMG_COLS = 160, 160
IMG_CHANNELS = 1


class DQNAgent:
    def __init__(self, learning_rate):
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding="same",
                         input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(9))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        print("We finish building the model")
        return model
