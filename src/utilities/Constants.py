from os import path
import socket

import tempfile

DATASETS = path.join('..', 'DataSets')

TEMP = tempfile.gettempdir()

DATASETS_MARKET = path.join(DATASETS, 'Market')

PROCESSED_LEXICONS = path.join(DATASETS, 'lexicons/')

TRAINING_BATCH = 10
TESTING_BATCH = 10
