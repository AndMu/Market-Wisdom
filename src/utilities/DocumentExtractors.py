import itertools
import os
from io import open

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from utilities import logger, Constants


class SingleFileLineSentence(object):

    def __init__(self, lexicon, source, max_sentence_length=10000, limit=None):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or no clipped if limit is None, the default).

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.lexicon = lexicon
        self.total = 0

    def __len__(self):
        return self.total

    def __iter__(self):
        self.total = 0
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                result = self.lexicon.review_to_sentences(utils.to_unicode(line))
                for sentence in result:
                    if self.total % 10000 == 0:
                        logger.info('Processed %i sentences', self.total)
                    self.total += 1
                    yield sentence
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    result = self.lexicon.review_to_sentences(utils.to_unicode(line))
                    for sentence in result:
                        if self.total % 10000 == 0:
                            logger.info('Processed %i sentences', self.total)
                        self.total += 1
                        yield sentence


class MultiFileLineSentence(object):

    def __init__(self, lexicon, source):
        self.lexicon = lexicon
        self.source = source
        self.total = 0
        self.totalFiles = 0

    def __len__(self):
        return self.total

    def __iter__(self):
        self.total = 0
        for root, subdirs, files in os.walk(self.source):
            for fname in files:
                full_path = os.path.join(root, fname)
                if '.bin' in full_path or '.npy' in full_path:
                    logger.debug('Ignore %s', fname)
                else:
                    self.totalFiles += 1
                    if self.totalFiles % 10000 == 0:
                        logger.info('Processed %i files and %i sentences', self.totalFiles, self.total)
                    with open(full_path, encoding='utf8') as file:
                        try:
                            text = file.read()
                            text = text.replace('\n', '')
                            result = self.lexicon.review_to_sentences(utils.to_unicode(text))
                            for sentence in result:
                                self.total += 1
                                yield sentence
                        except:
                            logger.error("failed processing file: %s", fname)
        logger.info('Processed %i', self.total)


class MultiFileLineDocument(object):

    def __init__(self, lexicon, source):
        self.lexicon = lexicon
        self.source = source

    def __iter__(self):
        for root, subdirs, files in os.walk(self.source):
            for fname in files:
                full_path = os.path.join(root, fname)
                with open(full_path, encoding='utf8') as file:
                    try:
                        text = file.read()
                        text = text.replace('\n', '')
                        words = self.lexicon.review_to_wordlist(utils.to_unicode(text))
                        tag = Constants.extract_tag(full_path)
                        tags = [tag]
                        yield TaggedDocument(words=words, tags=tags)
                    except:
                        logger.error("failed processing file: %s", fname)

