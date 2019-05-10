import logging.config
from sklearn.utils import shuffle

import numpy as np
from os import path
import quandl
from keras.layers import LSTM, Dropout, Dense, Activation
from keras import Sequential, callbacks
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from DataLoader import DataLoader
from MarketData import QuandlMarketDataSource, RedditMarketDataSource, BloombergMarketDataSource

from learning.BasicLearning import RbfClassifier
from utilities import Constants
from utilities.Utilities import Utilities

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
quandl.ApiConfig.api_key = '__YOUR_KEY__'


def build_model(inputs, model_type):

    model = Sequential()
    if model_type == 'Basic_LSTM':
        model.add(LSTM(400, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=False))
        model.add(Dropout(0.5))
    elif model_type == 'Conv':
        model.add(LSTM(200, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100,
                       return_sequences=False))
        model.add(Dropout(0.2))

    elif model_type == 'LSTM':
        model.add(LSTM(400, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(400, return_sequences=False))
        model.add(Dropout(0.3))
    else:
        raise ValueError(model_type)

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
    return model


def get_data(full_articles, sentiment_location, price_source, stock):
    loader = DataLoader()
    if price_source == 'quandl':
        source = QuandlMarketDataSource()
    elif price_source == 'reddit':
        source = RedditMarketDataSource()
    else:
        source = BloombergMarketDataSource()
    x_data, y_data = loader.load_data(stock, 5,
                                      source=source,
                                      sentiment_location=sentiment_location,
                                      full_articles=full_articles,
                                      from_date='2011-04-01', to_date='2015-04-01')
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)
    return x_data, y_data


def lstm_prediction(model_type, x_train, x_test, y_train):

    y_train = Utilities.make_dual(y_train, 2)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # initialise model architecture
    market_model = build_model(x_train, model_type)
    market_model.summary()
    # train model on data
    # note: eth_history contains information on the training error per epoch
    cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=25)]
    market_model.fit(x_train, y_train, batch_size=1000, callbacks=cbks, epochs=50, validation_split=0.25, shuffle=True)
    y_result_prob = market_model.predict(x_test)
    y_result = Utilities.make_single_dimension(y_result_prob)
    return y_result, y_result_prob


def svm_prediction(x_train, x_test, y_train):
    pipeline = Pipeline([
        ['clf', RbfClassifier()]])
    pipeline.fit(x_train, y_train)
    y_result = pipeline.predict(x_test)
    y_result_prob = pipeline.predict_proba(x_test)
    return y_result, y_result_prob


def processing(price_source, stock, load_articles, full_articles, processing_type='SVM'):

    x_data, y_data = get_data(full_articles, load_articles, price_source, stock)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

    if processing_type == 'SVM':
        y_result, y_result_prob = svm_prediction(x_train, x_test, y_train)
    else:
        y_result, y_result_prob = lstm_prediction(processing_type, x_train, x_test, y_train)

    Utilities.measure_performance(y_test, y_result)
    Utilities.measure_performance_auc(y_test, y_result, y_result)


if __name__ == '__main__':
    item = 'JPM'
    price_source = 'quandl'
    processing_type = 'SVM'
    sentiment_location = path.join(Constants.DATASETS_MARKET, 'Twitter/psenti.csv')
    # technical analysis
    processing(price_source, item, None, False, processing_type)
    # technical analysis + Sentiment
    sentiment_location = path.join(Constants.DATASETS_MARKET, 'FinArticles/psenti/all.results.csv')
    processing(price_source, item, sentiment_location, False, processing_type)
    sentiment_location = path.join(Constants.DATASETS_MARKET, 'FinArticles/psenti/reddit.results.csv')
    processing(price_source, item, sentiment_location, False, processing_type)
    # technical analysis + Sentiment + Mood
    processing(price_source, item, sentiment_location, True, processing_type)




