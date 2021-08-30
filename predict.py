import configparser

import numpy as np

from engins.model.nre import NREModel
import tensorflow as tf
from tensorflow_addons.text import crf_decode


class Predictor():
    def __init__(self, data_processor):
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.data_processor = data_processor
        self.model_type = config.get('ner', 'ner_model')
        self.checkpoint_name = config.get('ner', 'checkpoint_name')
        self.checkpoints_dir = config.get('ner', 'checkpoints_dir')
        self.max_to_keep = config.getint('ner', 'max_to_keep')
        self.model = NREModel(data_processor)

        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=self.checkpoints_dir,
                                                        max_to_keep=self.max_to_keep,
                                                        checkpoint_name=self.checkpoint_name)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    def predict(self, sentence):
        sentence_vector = self.data_processor.sentence_to_vector(sentence)
        sentence_vector = np.array([sentence_vector])
        logits = self.model(sentence_vector, training=False)
        print(logits)
        pred = tf.sigmoid(logits)

        self.extract_entity(pred, sentence)

    def extract_entity(self, pred, sentence):
        print(pred)
        start_index = np.where(pred[:, :, 0] > 0.5)
        end_index = np.where(pred[:, :, 0] < 0.5)
        print('start', start_index)
        print('end', end_index)
        if len(start_index) > len(end_index):
            for i in end_index:
                print(sentence[start_index[i]:end_index[i]])
        else:
            for i in start_index:
                print(sentence[start_index[i]:end_index[i]])
