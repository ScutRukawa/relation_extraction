import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd
import csv
import configparser


class DataProcessor():
    def __init__(self):
        super(DataProcessor, self).__init__()
        config = configparser.ConfigParser()
        config.read('./config/config.ini')
        self.max_seq_length = config.getint('ner', 'max_sequence_length')
        if not os.path.exists('./data/token2id.csv'):
            self.build_voc()
        predicate = pd.read_csv('./data/predicate_class.csv', sep=' ')
        token2id_data = pd.read_csv('./data/token2id.csv', sep=' ', quoting=csv.QUOTE_NONE)
        entity = pd.read_csv('./data/entity_class.csv', sep=' ')
        self.predicate_classes = dict(zip(list(predicate.label), list(predicate.id)))
        self.token2id = dict(zip(list(token2id_data.token), list(token2id_data.id)))
        self.id2token = dict(zip(list(token2id_data.id), list(token2id_data.token)))
        self.class2id = dict(zip(list(entity.label), list(entity.id)))
        self.id2class = dict(zip(list(entity.id), list(entity.label)))
        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'
        # print(self.token2id)
        self.X = np.array([])
        self.y = np.array([])

    def build_voc(self):
        voc = {}
        spo_dict = {}
        entity_dict = {}
        with open('./data/train_data.json', encoding='utf-8') as train_file:
            for line in train_file:
                train_data_line = json.loads(line)
                for token in train_data_line['text']:
                    voc[token] = 0
                for spo in train_data_line['spo_list']:
                    spo_dict[spo['predicate']] = 0
                    entity_dict[spo['object_type']] = 0
                    entity_dict[spo['subject_type']] = 0
        with open('./data/token2id.csv', 'w', encoding='utf-8') as token2id_file:
            token2id_file.write('token id\n')
            count = 0
            for token in voc.keys():
                if token != ' ':
                    token2id_file.write(token + ' ' + str(count) + '\n')
                    count += 1

            token2id_file.write('[PAD] ' + str(count) + '\n')
            token2id_file.write('[UNK] ' + str(count + 1) + '\n')
        with open('./data/predicate_class.csv', 'w', encoding='utf-8') as predicate_file:
            predicate_file.write('label id\n')
            count = 0
            for spo in spo_dict.keys():
                predicate_file.write(spo + ' ' + str(count) + '\n')
                count += 1
        with open('./data/entity_class.csv', 'w', encoding='utf-8') as entity_file:
            entity_file.write('label id\n')
            count = 0
            for entity in entity_dict.keys():
                entity_file.write(entity + ' ' + str(count) + '\n')
                count += 1

    def get_dataset(self, file_name):
        sentences, pegs = [], []
        entity_class_nums = len(self.class2id)
        # './data/train_data.json'
        with open(file_name, encoding='utf-8') as train_file:
            for line in train_file:
                train_data_line = json.loads(line)
                text = train_data_line['text']
                if len(text) > self.max_seq_length:
                    text = text[0:self.max_seq_length]
                peg = np.zeros((self.max_seq_length, entity_class_nums, 2))
                for items in train_data_line['spo_list']:
                    subject_type = self.class2id[items['subject_type']]
                    object_type = self.class2id[items['object_type']]
                    start_index_subject = text.find(items['subject'])
                    start_index_object = text.find(items['object'])

                    if start_index_subject != -1 and start_index_object != -1:
                        end_index_subject = start_index_subject + len(items['subject']) - 1
                        end_index_object = start_index_object + len(items['object']) - 1
                        # onehot 编码分类
                        peg[start_index_subject, subject_type, 0] = 1
                        peg[end_index_subject, subject_type, 1] = 1
                        peg[start_index_object, object_type, 0] = 1
                        peg[end_index_object, object_type, 1] = 1
                sentence_vector = self.sentence_to_vector(text)
                sentences.append(self.padding(sentence_vector))
                pegs.append(peg)
        return np.array(sentences[0:30000]), np.array(pegs[0:30000])

    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_seq_length:
            for _ in range(self.max_seq_length - len(sentence)):
                sentence.append(self.token2id[self.PADDING])

        else:
            sentence = sentence[:self.max_seq_length]
        return sentence

    def sentence_to_vector(self, sentence):
        vector = []
        cut_words = str(sentence).strip()
        for token in cut_words:
            if token in self.token2id:
                vector.append(self.token2id[token])
            else:
                vector.append(self.token2id[self.UNKNOWN])
        vector = self.padding(vector)
        return vector

    def vector_to_sentence(self, vector):
        sentence = []
        for id in vector:
            if id in self.id2token and id!=self.token2id[self.PADDING]:
                sentence.append(self.id2token[id])
            else:
                sentence.append(self.UNKNOWN)
        return sentence
