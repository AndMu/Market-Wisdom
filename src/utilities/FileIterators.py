import abc
import re

import shutil


from pathlib2 import Path

import numpy as np
from os import path, walk, makedirs

from utilities import Constants, logger
import io

from utilities.NumpyHelper import NumpyDynamic


class FileIterator(object):
    def __init__(self, source, data_path):
        self.source = source
        self.data_path = data_path

    def __iter__(self):
        logger.info("Loading %s...", self.data_path)

        for (root, dir_names, files) in walk(self.data_path):
            for name in files:
                file_name = path.join(root, name)
                data = self.source.get_vector(file_name)
                yield data, name


class DataIterator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, source, root, data_path):
        split_result = path.split(data_path)
        if len(split_result) > 1:
            self.name = path.split(data_path)[1]
        else:
            self.name = data_path
        self.source = source
        self.data_path = path.join(root, data_path)
        root_name = path.split(root)[1]
        sub_folder = ''.join(ch for ch in data_path if ch.isalnum())
        self.bin_location = path.join(Constants.TEMP, 'bin', root_name, sub_folder, self.source.word2vec.name)

    @abc.abstractmethod
    def __iter__(self):
        pass

    def delete_cache(self):
        if Path(self.bin_location).exists():
            logger.info('Deleting [%s] cache dir', self.bin_location)
            shutil.rmtree(self.bin_location)

    def get_data(self):
        all_data_path = path.join(self.bin_location, 'all')
        if not Path(all_data_path).exists():
            makedirs(all_data_path)

        data_file = Path(all_data_path + '_data.npy')
        class_file = Path(all_data_path + '_class.npy')
        name_file = Path(all_data_path + '_name.npy')
        if data_file.exists():
            logger.info('Found created file. Loading %s...', str(data_file))
            data = np.load(str(data_file))
            type_data = np.load(str(class_file))
            names_data = np.load(str(name_file))
            logger.info('Using saved data %s with %i records', str(data_file), len(data))
            return data, names_data, type_data

        vectors = NumpyDynamic(np.object)
        values = NumpyDynamic(np.int32)
        file_names = NumpyDynamic(np.object)
        length = []
        for item_class, name, item in self:
            vectors.add(item)
            file_names.add(name)
            values.add(item_class)
            length.append(len(item))

        data = vectors.finalize()
        names_data = file_names.finalize()
        type_data = values.finalize()

        if len(data) == 0:
            raise StandardError("No files found")
        total =(float(len(length) + 0.1))
        logger.info("Loaded %s - %i with average length %6.2f, min: %i and max %i", self.data_path, len(data),
                    sum(length) / total, min(length), max(length))
        logger.info('Saving %s', str(data_file))
        np.save(str(data_file), data)
        np.save(str(class_file), type_data)
        np.save(str(name_file), names_data)
        return data, names_data, type_data


class ClassDataIterator(DataIterator):

    def __iter__(self):
        pos_files = FileIterator(self.source, path.join(self.data_path, 'pos'))
        neg_files = FileIterator(self.source, path.join(self.data_path, 'neg'))

        for vector, name, in pos_files:
            yield 1, name, vector
        for vector, name in neg_files:
            yield 0, name, vector


class SingeDataIterator(DataIterator):

    def __iter__(self):
        pos_files = FileIterator(self.source, self.data_path)
        for vector, name in pos_files:
            yield -1, name, vector


class SemEvalFileReader(object):
    def __init__(self, file_name, source, convertor):
        self.file_name = file_name
        self.source = source
        self.convertor = convertor

    def __iter__(self):
        with io.open(self.file_name, 'rt', encoding='utf8') as csv_file:
            logger.info('Loading: %s', self.file_name)
            for line in csv_file:
                row = re.split(r'\t+', line)
                review_id = row[0]
                total_rows = len(row)
                if total_rows >= 3:
                    type_class = self.convertor.is_supported(row[total_rows - 2])
                    if type_class is not None:
                        text = row[total_rows - 1]
                        vector = self.source.get_vector_from_review(text)
                        if vector is not None:
                            yield type_class, review_id, vector
                        else:
                            logger.warn("Vector not found: %s", text)


class SemEvalDataIterator(DataIterator):

    def __init__(self, source, root, data_path, convertor):
        super(SemEvalDataIterator, self).__init__(source, root, data_path)
        self.bin_location += convertor.name
        self.convertor = convertor

    def __iter__(self):
        if path.isfile(self.data_path):
            for type_class, review_id, vector in SemEvalFileReader(self.data_path, self.source, self.convertor):
                yield type_class, review_id, vector
        else:
            for (root, dir_names, files) in walk(self.data_path):
                for name in files:
                    file_name = path.join(root, name)
                    for type_class, review_id, vector in SemEvalFileReader(file_name, self.source, self.convertor):
                        yield type_class, review_id, vector



