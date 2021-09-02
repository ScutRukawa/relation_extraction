import numpy as np
import json
import pandas as pd
import tensorflow as tf


def extract_entity(model, sentence, data_processor):
    entity_result = {}
    entity_class = pd.read_csv('./data/entity_class.csv', sep=' ')
    for label in entity_class.label:
        entity_result[label] = set()
    sentence_vector = data_processor.sentence_to_vector(sentence)
    sentence_tensor = tf.expand_dims(sentence_vector, axis=0)
    pred = model(sentence_tensor)
    start_array = pred[0][:, :, 0]
    end_array = pred[0][:, :, 1]
    start = np.where(start_array > 0.5)
    end = np.where(end_array > 0.5)
    for _start, entity_type_start in zip(*start):
        for _end, entity_type_end in zip(*end):
            if _end >= _start and entity_type_start == entity_type_end:
                entity = sentence[int(_start):int(_end) + 1]
                entity_result.setdefault(data_processor.id2class[entity_type_start], set()).add(entity)
                break

    return entity_result


def evaluate(model, data_processor):
    counts = {}
    metrics_entity = {}
    entity_class = pd.read_csv('./data/entity_class.csv', sep=' ')
    for label in entity_class.label:
        counts[label] = {}
        counts[label]['predict_TP'] = 0.
        counts[label]['predict_count'] = 0.
        counts[label]['true_count'] = 0.
        metrics_entity[label] = {}
        metrics_entity[label]['f1'] = 0.
        metrics_entity[label]['precision'] = 0.
        metrics_entity[label]['recall'] = 0.
    with open('./data/dev_data.json', encoding='utf-8') as dev_data_file:
        count_text = 1
        for line in dev_data_file:
            count_text += 1
            if count_text >= 1000:
                break
            train_data_line = json.loads(line)
            entity_set_pred = extract_entity(model, train_data_line['text'], data_processor)
            entity_set_true = {}
            for label in entity_class.label:
                entity_set_true[label] = set()
            for spo in train_data_line['spo_list']:
                entity_set_true.setdefault(spo['object_type'], set()).add(spo['object'])
                entity_set_true.setdefault(spo['subject_type'], set()).add(spo['subject'])
            for label in counts.keys():
                counts[label]['predict_TP'] += len(entity_set_pred[label] & entity_set_true[label])
                counts[label]['predict_count'] += len(entity_set_pred[label])
                counts[label]['true_count'] += len(entity_set_true[label])
    for label, count in counts.items():
        f1, precision, recall = 2 * count['predict_TP'] / (count['true_count'] + count['predict_count'] + 1e-5), count[
            'predict_TP'] / (count['true_count'] + 1e-5), count['predict_TP'] / (count['predict_count'] + 1e-5)
        metrics_entity[label]['f1'] = f1
        metrics_entity[label]['precision'] = precision
        metrics_entity[label]['recall'] = recall
    return metrics_entity
