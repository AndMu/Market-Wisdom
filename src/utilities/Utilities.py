import abc
import json
from os import path

import _pickle as cPickle
from sklearn import metrics

import numpy as np
from sklearn.metrics import roc_auc_score

from utilities import logger


class Utilities(object):
    @staticmethod
    def load_json(fname):
        logger.info("load_json - %s", fname)
        with open(fname) as f:
            return json.loads(f.read())

    @staticmethod
    def lines(file_name):
        with open(file_name) as f:
            for line in f:
                yield line

    @staticmethod
    def load_pickle(file_name):
        with open(file_name) as f:
            return cPickle.load(f)

    @staticmethod
    def make_single_dimension(y):
        y = np.copy(y)
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_value = np.argmax(y, axis=1)
        else:
            y[y > 0.5] = 1
            y[y < 0.5] = 0
            y_value = y.astype(int)
            if len(y.shape) > 1 and y.shape[1] == 1:
                y_value = y_value[:, 0]

        return y_value

    @staticmethod
    def save_pickle(file_name, data):
        with open(file_name, 'wb') as f:
            return cPickle.dump(data, f)

    @staticmethod
    def make_dual(y, number_classes):
        y_dual = np.zeros((y.shape[0], number_classes), dtype=y.dtype)
        index = 0
        for class_item in range(0, number_classes):
            y_dual[y == class_item, index] = 1
            index += 1
        return y_dual

    @staticmethod
    def create_vector(x, max_words):
        x_result = np.zeros((x.shape[0], max_words), dtype=x.dtype)
        for i in range(0, x.shape[0]):
            for record in x[i]:
                if 0 < record < max_words:
                    x_result[i, record - 1] += 1
        return x_result


    @staticmethod
    def make_binary_prob(y):
        y_dual = np.zeros((y.shape[0], 2))
        y = np.reshape(y, y.shape[0])
        y_dual[:, 0] = 1 - y
        y_dual[:, 1] = y
        return y_dual

    @staticmethod
    def measure_performance(test_y, result_y):
        report = metrics.classification_report(test_y, result_y, digits=3)
        logger.info('\n{}'.format(report))

        if len(np.unique(test_y)) != 1:
            macro = metrics.f1_score(test_y, result_y, average='macro')
            logger.info('Macro F1 {0:.3f}'.format(macro))

            macro = metrics.f1_score(test_y, result_y, average='micro')
            logger.info('Micro F1 {0:.3f}'.format(macro))

    @staticmethod
    def measure_performance_auc(test_y, result_y, result_y_prob):
        try:
            vacc = metrics.accuracy_score(test_y, result_y)
            # find validation AUC
            if len(np.unique(test_y)) == 2:
                vauc = roc_auc_score(test_y, result_y_prob)
                logger.info('Accurary: {0:.3f} and AUC {1:.3f}'.format(vacc, vauc))
            else:
                vauc = None
                logger.info('Accurary: {0:.3f}'.format(vacc))

            return vacc, vauc
        except:
            logger.error("Error calculating metrics")

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @staticmethod
    def count_occurences(y):
        if len(y.shape) > 1 and y.shape[1] > 1:
            for current in range(0, y.shape[1]):
                total = sum(y[:, current])
                logger.info("Found %s type with %i records", current, total)
        else:
            types = np.unique(y)
            for current in types:
                total = sum(y == current)
                logger.info("Found %s type with %i records", current, total)


class ClassConvertor(object):

    ignore_error = False

    def __init__(self, name, class_dict):
        self.class_dict = class_dict
        self.classes = np.unique(class_dict.values())
        self.name = "{}_{}".format(name, len(self.classes))

    def total_classes(self):
        return len(self.classes)

    def create_vector(self, y):
        if self.total_classes() > 2:
            return Utilities.make_dual(y, self.total_classes())
        return y

    def is_binary(self):
        return self.total_classes() == 2

    def is_supported(self, y):
        if y not in self.class_dict:
            if not ClassConvertor.ignore_error:
                logger.warn("Value %s not supported", y)
            return None
        return self.class_dict[y]

    def make_single(self, y):
        if self.total_classes() == 2:
            if len(y.shape) > 1 and y.shape[1] > 1:
                return y[:, 1]
            if len(y.shape) > 1 and y.shape[1] == 1:
                return y[:, 0]
            return y
        else:
            return np.amax(y, axis=1)


class CollectionUtilities(object):
    """General collection related utilities"""

    @staticmethod
    def make_unique_list(input_list):
        seen = set()
        result = [item for item in input_list if item not in seen and not seen.add(item)]
        return result

    @staticmethod
    def make_unique_tuple_list(input_list):
        seen = set()
        result = [item for item in input_list if item[0] not in seen and not seen.add(item[0])]
        return result

    @staticmethod
    def get_words_list(input_list):
        return list(set([x[0].lower() for x in input_list]))