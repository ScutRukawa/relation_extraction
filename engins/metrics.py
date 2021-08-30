import numpy as np

from data import DataProcessor


def get_pred_index(pred):
    batch_size = pred[0]

    for i in range(batch_size):
        sentence_index = pred[i]
        start_index = 0
        end_index = 0
        for j in pred[0][0]:
            if pred[i][j] == 1 and start_index == 0:
                start_index = j
            elif pred[i][j] == 1 and start_index != 0:
                end_index = j


def extract_entity(pred, X, data_processor):
    batch_size = pred.shape[0]
    seq_length = pred.shape[1]
    class_num = pred.shape[2]
    entity_set = []
    for batch_index in range(batch_size):
        sentence_ids = X[batch_index]
        sentence = data_processor.vector_to_sentence(sentence_ids.numpy())
        start_array = pred[batch_index][:, :, 0]
        end_array = pred[batch_index][:, :, 1]
        start = np.where(start_array > 0.5)
        end = np.where(end_array > 0.5)
        for _start, entity_type_start in zip(*start):
            for _end, entity_type_end in zip(*end):
                if _end >= _start and entity_type_start == entity_type_end:
                    entity = sentence[int(_start):int(_end) + 1]
                    entity_set.append([int(entity_type_start), str(entity)])
                    break
    return entity_set
