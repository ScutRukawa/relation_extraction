import configparser
from data import DataProcessor
from tensorflow.keras import optimizers, metrics
from tqdm import tqdm
from engins.model.nre import NREModel
import tensorflow as tf
from tensorflow_addons.text import crf_decode
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from engins.metrics import extract_entity


class TrainProcessor:
    def __init__(self):
        super(TrainProcessor, self).__init__()
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        if config.get('ner', 'optimizers') == 'Adam':
            self.optimizers = optimizers.Adam(lr=0.0001)
        else:
            self.optimizers = optimizers.RMSprop(lr=0.0001)
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
            # 重新打乱
            sample_num = len(X_train)
            train_batch = tf.data.Dataset.from_tensor_slices((X_train[0:5000], y_train[0:5000])).shuffle(
                sample_num).batch(
                batch_size=self.batch_size)
            loss_func = BinaryCrossentropy(from_logits=True)
            for step, (X, y) in tqdm(train_batch.enumerate(), desc='epoch:' + str(epoch)):
                with tf.GradientTape() as tape:
                    logits = model(X, training=True)
                    loss = loss_func(logits, y)
                grads = tape.gradient(loss, model.trainable_variables)
                self.optimizers.apply_gradients(zip(grads, model.trainable_variables))

            count = 0.0
            loss = 0.
            dev_batch = tf.data.Dataset.from_tensor_slices((X_dev[0:100], y_dev[0:100])).batch(
                batch_size=self.batch_size)
            for step, (X, y) in dev_batch.enumerate():
                count += 1
                logits = model(X, training=False)
                pred = tf.sigmoid(logits)
                entity_set = extract_entity(pred, X, data_processor)
                if len(entity_set[1])!=0:
                    for entity in entity_set:
                        print(entity)
                loss += loss_func(logits, y)

            print("epoch %d : %f" % (epoch, loss / count))
            checkpoint_manager.save()
