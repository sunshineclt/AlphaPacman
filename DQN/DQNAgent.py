from keras.initializers import random_normal
from keras.layers import Conv2D, MaxPool2D
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, learning_rate, img_rows, img_cols, img_channels, initialize_stddev):
        self.model = self.build_model(learning_rate, img_rows, img_cols, img_channels, initialize_stddev)
        self.target_model = self.build_model(learning_rate, img_rows, img_cols, img_channels, initialize_stddev)
        self.step = 0

    @staticmethod
    def build_model(learning_rate, img_rows, img_cols, img_channels, initialize_stddev):
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same",
                         input_shape=(img_rows, img_cols, img_channels),
                         kernel_initializer=random_normal(stddev=initialize_stddev)))
        model.add(PReLU())
        model.add(Conv2D(32, (4, 4), strides=(2, 2), padding='same',
                         kernel_initializer=random_normal(stddev=initialize_stddev)))
        model.add(PReLU())
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(256, kernel_initializer=random_normal(stddev=initialize_stddev)))
        model.add(PReLU())
        model.add(Dense(9, kernel_initializer=random_normal(stddev=initialize_stddev)))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        print("We finish building the model")
        return model

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
