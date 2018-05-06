import re

from nltk import PorterStemmer


class TextHelper(object):
    """Text utility class"""

    stemmer = PorterStemmer()
    stem_words = False

    @staticmethod
    def get_raw_word(word):
        word = word.lower()
        if TextHelper.is_emoticon(word) or TextHelper.is_hash(word):
            return word

        if TextHelper.stem_words:
            return TextHelper.stemmer.stem(word)

        return word

    @staticmethod
    def is_emoticon(word):
        return 'emoticon_' in word

    @staticmethod
    def is_hash(word):
        return len(word) > 1 and word[0] == '#'

    @staticmethod
    def is_usable_word(word):
        result = re.match('^[a-zA-Z][a-zA-Z0-9]{2,}$', word)
        return result is not None

    @staticmethod
    def is_simple_word(word):
        result = re.match('^[a-zA-Z]{3,}$', word)
        return result is not None

    @staticmethod
    def is_match(word_1, word_2):
        return word_1.lower() == word_2.lower()

    @staticmethod
    def is_list_match(word, data_list):
        return any(TextHelper.is_match(word, item) for item in data_list)


