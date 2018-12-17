import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.models import Model, load_model, Sequential

import tflearn
from tflearn import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression


learning_rate = 1e-3


def tflearn_walker_model(training_shape):

    walker = input_data(shape=[None, training_shape, 1], name='input')

    #walker = fully_connected(walker, 3, activation='relu')

    walker = fully_connected(walker, 64, activation='relu')
    walker = dropout(walker, 0.75)

    walker = fully_connected(walker, 128, activation='relu')
    walker = dropout(walker, 0.75)

    walker = fully_connected(walker, 256, activation='relu')
    walker = dropout(walker, 0.75)

    walker = fully_connected(walker, 128, activation='relu')
    walker = dropout(walker, 0.75)

    walker = fully_connected(walker, 64, activation='relu')
    walker = dropout(walker, 0.75)

    walker = fully_connected(walker, 3, activation='softmax')
    walker = regression(walker, optimizer='adam', loss='categorical_crossentropy', learning_rate=learning_rate, name='tiles')

    model = tflearn.DNN(walker, tensorboard_dir='/home/sinandeger/tensorboard_logs/')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = tflearn_walker_model(len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='walking')
    return model


def keras_walker_model(training_shape):

    X_input = Input(training_shape)

    move_decision = Sequential()

    move_decision.add(Dense(units=3, input_dim=training_shape, activation='relu', name='input+hidden'))(X_input)

    return move_decision


def keras_conv_walker(training_shape):

    board = Input(training_shape)

    conv1 = Conv2D(8, (2, 2), activation='relu', padding='none')(board)
    conv2 = Conv2D(16, (2, 2), activation='relu', padding='none')(conv1)
    conv3 = Conv2D(32, (2, 2), activation='relu', padding='none')(conv2)
    conv4 = Conv2D(16, (2, 2), activation='relu', padding='none')(conv3)
    conv5 = Conv2D(8, (2, 2), activation='relu', padding='none')(conv4)

    output_board = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    conv_walker = Model(inputs=board, outputs=output_board, name='Keras-Conv-Walker')
    return conv_walker

