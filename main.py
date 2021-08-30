import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random
from data import DataProcessor
from tensorflow.keras.losses import BinaryCrossentropy, categorical_crossentropy, CategoricalCrossentropy, \
    binary_crossentropy

import math
from data import DataProcessor
from engins.train import TrainProcessor
from predict import Predictor

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainProcessor = TrainProcessor()
    trainProcessor.train()
    res = []
    test = ([111, 222], [333, 444])
    for x, y in zip(*test):
        print(x, y)
        res.append([x, ['y']])
    print(res)