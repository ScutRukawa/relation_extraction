from abc import ABC
import configparser
import tensorflow as tf
import tensorflow.keras.backend as K
from data import DataProcessor

from tensorflow.keras import regularizers


class NREModel(tf.keras.Model, ABC):
    def __init__(self, data_processor):
        super(NREModel, self).__init__()
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.max_seq_length = config.getint('ner', 'max_sequence_length')
        self.class_num = len(data_processor.predicate_classes)
        self.entity_class_num = len(data_processor.class2id)
        self.voc_size = 8180
        self.batch_size = config.getint('ner', 'batch_size')
        self.is_early_stop = config.getboolean('ner', 'is_early_stop')
        self.patient = config.getint('ner', 'patient')
        self.embedding_method = config.get('ner', 'embedding_method')
        self.embedding_dim = config.getint('ner', 'embedding_dim')
        self.embedding = tf.keras.layers.Embedding(input_length=self.max_seq_length, input_dim=self.voc_size + 1,
                                                   output_dim=self.embedding_dim)
        # pointer model
        self.hidden_dim = config.getint('ner', 'hidden_dim')
        self.bilstm = tf.keras.layers.LSTM(units=self.hidden_dim, activation='tanh', return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=2 * self.entity_class_num, activation=None,
                                           kernel_regularizer=regularizers.l2(0.001))
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def position_embedding(self):
        # 生成sin cos
        pid = tf.range(self.max_seq_length, dtype=tf.float32)
        div_term = 1.0 / (tf.pow(10000, 2 * tf.range(self.embedding_dim // 2, dtype=tf.float32) / self.embedding_dim))
        position_i = tf.sin(pid * div_term)
        position_j = tf.cos(pid * div_term)

        # 合并sin cos
        pos_embedding = tf.constant([self.embedding_dim])
        indices_i = tf.expand_dims(tf.range(0, self.embedding_dim, 2), 1)
        indices_j = tf.expand_dims(tf.range(1, self.embedding_dim, 2), 1)
        return tf.scatter_nd(indices_i, position_i, pos_embedding) + tf.scatter_nd(indices_j, position_j, pos_embedding)

    def get_pointer(self, input):
        index_count = 0
        flag = 0
        start = 0
        pointer = []
        for v in input.numpy():
            if v > 0.5:
                if flag % 2 != 0:
                    start = index_count
                    flag = 0
                else:
                    end = index_count
                    pointer.append((start, end))
                    flag = 1
        index_count += 1

        return pointer

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        embedding = self.layer_norm(embedding)
        bilstm_out = self.bilstm(embedding)
        # flatten_out = self.flatten(bilstm_out)
        batch_size = inputs.shape[0]
        logits = self.dense(bilstm_out)  # [batch_size,seq_length,class]
        output = tf.nn.sigmoid(logits)
        output = tf.reshape(output, [batch_size, -1, self.entity_class_num, 2])
        return output
