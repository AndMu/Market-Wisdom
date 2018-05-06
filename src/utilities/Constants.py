from os import path
import socket


hostname = socket.gethostname()

if hostname.lower() == 'main-pc':
    ROOT_LOCATION = 'e:/'
elif hostname.lower() == 'dev-pc':
    ROOT_LOCATION = '//storage/monitoring'
else:
    ROOT_LOCATION = 'c:/'




DATASETS = path.join(ROOT_LOCATION, 'DataSets')
TEMP = "C:/Temp/Sentiment"

DATASETS_MARKET = path.join(DATASETS, 'Market')

PROCESSED_LEXICONS = path.join(DATASETS, 'lexicons/')

TRAINING_BATCH = 10
TESTING_BATCH = 10
