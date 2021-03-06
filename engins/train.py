import configparser
import json

from data import DataProcessor
from tensorflow.keras import optimizers, metrics
from tqdm import tqdm
from engins.model.nre import NREModel
import tensorflow as tf
from tensorflow_addons.text import crf_decode
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from engins.metrics import evaluate
from tensorflow_addons.losses import sigmoid_focal_crossentropy


class TrainProcessor:
    def __init__(self):
        super(TrainProcessor, self).__init__()
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        if config.get('ner', 'optimizers') == 'Adam':
            self.optimizers = optimizers.Adam(lr=0.01)
        else:
            self.optimizers = optimizers.RMSprop(lr=0.01)
        self.epochs = config.getint('ner', 'epochs')
        self.batch_size = config.getint('ner', 'batch_size')
        self.max_sequence_length = config.getint('ner', 'max_sequence_length')
        self.PADDING = config.get('ner', 'PADDING')
        self.UNKNOWN = config.get('ner', 'UNKNOWN')
        self.is_early_stop = config.getboolean('ner', 'is_early_stop')
        self.patient = config.getint('ner', 'patient')
        self.checkpoints_dir = config.get('ner', 'checkpoints_dir')
        self.max_to_keep = config.getint('ner', 'max_to_keep')
        self.checkpoint_name = config.get('ner', 'checkpoint_name')
        self.stop_words = config.get('ner', 'stop_words')
        self.embedding_method = config.get('ner', 'embedding_method')

    def train(self):
        data_processor = DataProcessor()
        model = NREModel(data_processor)
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=self.checkpoints_dir,
                                                        max_to_keep=self.max_to_keep,
                                                        checkpoint_name=self.checkpoint_name)
        X_train, y_train = data_processor.get_dataset('./data/train_data.json')
        X_dev, y_dev = data_processor.get_dataset('./data/dev_data.json')
        for epoch in range(self.epochs):
            # ????????????
            sample_num = len(X_train)
            train_batch = tf.data.Dataset.from_tensor_slices((X_train[0:20000], y_train[0:20000])).shuffle(
                sample_num).batch(
                batch_size=self.batch_size)
            for step, (X, y) in tqdm(train_batch.enumerate(), desc='epoch:' + str(epoch)):
                with tf.GradientTape() as tape:
                    output = model(X, training=True)
                    y = tf.cast(y, dtype=tf.float32)
                    loss = sigmoid_focal_crossentropy(y, output)
                    loss_class = tf.reduce_mean(loss, axis=-1)
                    loss_sentence = tf.reduce_sum(loss_class, axis=-1)
                    loss_sum = tf.reduce_sum(loss_sentence, axis=-1)
                grads = tape.gradient(loss_sum, model.trainable_variables)
                self.optimizers.apply_gradients(zip(grads, model.trainable_variables))

            count = 0.0
            loss_sum = 0.
            dev_batch = tf.data.Dataset.from_tensor_slices((X_dev[0:1000], y_dev[0:1000])).batch(
                batch_size=self.batch_size)
            for step, (X, y) in dev_batch.enumerate():
                count += 1
                output = model(X, training=False)
                y = tf.cast(y, dtype=tf.float32)
                loss = sigmoid_focal_crossentropy(y, output)
                loss_class = tf.reduce_mean(loss, axis=-1)
                loss_sentence = tf.reduce_sum(loss_class, axis=-1)
                loss_sum += tf.reduce_sum(loss_sentence, axis=-1)
            metrics_entity = evaluate(model, data_processor)
            print("epoch %d : %f" % (epoch, loss_sum / count))
            print(metrics_entity)
            checkpoint_manager.save()

    def pprint(self, pred):
        return
