import abc

import logging

from utilities.FileIterators import ClassDataIterator, SingeDataIterator, SemEvalDataIterator
logger = logging.getLogger(__name__)


class DataLoader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parser, convertor, root):
        self.root = root
        self.convertor = convertor
        self.parser = parser

    @abc.abstractmethod
    def get_iterator(self, data_path, class_iter=True):
        pass

    def get_data(self, data_path, delete=False, class_iter=True):
        logger.info("Loading [%s]...", data_path)
        train_iterator = self.get_iterator(data_path, class_iter)

        if delete:
            train_iterator.delete_cache()

        data_x, name, data_y = train_iterator.get_data()
        return name, data_x, data_y


class ImdbDataLoader(DataLoader):

    def get_iterator(self, data_path, class_iter=True):
        if class_iter:
            return ClassDataIterator(self.parser, self.root, data_path)

        return SingeDataIterator(self.parser, self.root, data_path)


class SemEvalDataLoader(DataLoader):

    def __init__(self, parser, convertor, root):
        super(SemEvalDataLoader, self).__init__(parser, convertor, root)
        self.convertor = convertor

    def get_iterator(self, data_path, class_iter=True):
        return SemEvalDataIterator(self.parser, self.root, data_path, self.convertor)