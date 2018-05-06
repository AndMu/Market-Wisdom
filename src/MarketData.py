import numpy as np
import quandl
import pandas as pd
from os import path

from PortfolioBasic.Definitions import HeaderFactory
from utilities import Constants, logger



class MarketData(object):

    def __init__(self, stock_name, df, ma=[], days=5, indicator=None, df_additional=None):
        self.scalers = {}
        self.stock_name = stock_name,
        self.df = df
        self.ma = ma
        self.indicator = indicator
        self.df_additional = df_additional
        self.days = days

    def get_binary_data(self, df, days=5, from_date='2014-04-01', to_date='2015-04-01'):

        df = df.ix[from_date:to_date].copy()
        df_full = pd.merge(df, self.df.ix[from_date:to_date], left_index=True, right_index=True, how='outer')
        df_full.to_csv('result.csv')
        y_data_df = df['Direction'].copy()
        # df.drop(labels=[HeaderFactory.Price], axis=1, inplace=True)
        df.drop(labels=['Direction',
                        'Open',
                        'High',
                        'Low',
                        'Volume',
                        'Close',
                        ], axis=1, inplace=True)

        x_data = df.iloc[1:-days, :].values
        y_data = y_data_df.iloc[1:-days].values

        df['y'] = y_data_df
        df.iloc[1:-days].to_csv('training.csv')
        return x_data, y_data

    def get_stock_data(self):
        df = self.df.copy()
        # Percentage change
        df['Pct'] = df[HeaderFactory.Price].pct_change()
        df['Direction'] = np.where(df[HeaderFactory.Price].shift(-self.days) <= df[HeaderFactory.Price], 0, 1)

        # Moving Average
        if self.ma != []:
            for moving in self.ma:
                df['{}ma'.format(moving)] = df[HeaderFactory.Price].rolling(window=moving).mean()

        indicator_columns = []
        if self.indicator is not None:
            df_signal = self.indicator.calculate(df)
            df = df.join(df_signal)
            indicator_columns = list(df_signal.columns.values)

        df.dropna(inplace=True)

        if self.df_additional is not None:
            df = df.join(self.df_additional)
            indicator_columns.extend(list(self.df_additional.columns.values))

        df.fillna(value=0, inplace=True)

        # Move Adj Close to the rightmost for the ease of training
        adj_close = df[HeaderFactory.Price]
        df.drop(labels=[HeaderFactory.Price], axis=1, inplace=True)
        df = pd.concat([df, adj_close], axis=1)
        return df


class QuandlMarketDataSource(object):

    def get_stock_data(self, stock_name):
        """
                Return a dataframe of that stock and normalize all the values.
                (Optional: create moving average)
                """
        logger.info("Loading Stock [%s]...", stock_name)
        df = quandl.get_table('WIKI/PRICES', ticker=stock_name, paginate=True)
        df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend', 'volume', 'split_ratio'], 1, inplace=True)
        df.set_index('date', inplace=True)

        # Renaming all the columns so that we can use the old version code
        df.rename(columns={'adj_open': 'Open', 'adj_high': 'High', 'adj_low': 'Low', 'adj_volume': 'Volume',
                           'adj_close': HeaderFactory.Price}, inplace=True)

        df.sort_index(ascending=True, inplace=True)
        df.dropna(inplace=True)
        return df


class RedditMarketDataSource(object):

    def get_stock_data(self, stock_name):
        file_path = path.join(Constants.DATASETS_MARKET, 'reddit/DJIA_table.csv')
        logger.info("Loading [%s]...", file_path)
        market_data = pd.read_csv(file_path, na_values=['nan'])
        # drop unadjusted close
        market_data.Date = pd.to_datetime(market_data.Date, format='%Y-%m-%d')
        market_data.set_index('Date', inplace=True)
        market_data.reindex()
        market_data.sort_index(ascending=True, inplace=True)
        market_data.drop(labels=[HeaderFactory.Price], axis=1, inplace=True)
        market_data.rename(columns={"Adj Close": HeaderFactory.Price}, inplace=True)
        market_data.dropna(inplace=True)
        return market_data


class BloombergMarketDataSource(object):

    def get_stock_data(self, stock_name: str):
        file_path = path.join(Constants.DATASETS_MARKET, 'stock/{}.csv'.format(stock_name))
        logger.info("Loading [%s]...", file_path)
        market_data = pd.read_csv(file_path, na_values=['nan'])
        # drop unadjusted close
        market_data.Date = pd.to_datetime(market_data.Date, format='%Y-%m-%d')
        market_data.set_index('Date', inplace=True)
        market_data.reindex()
        market_data.sort_index(ascending=True, inplace=True)
        if 'curncy' in stock_name.lower():
            market_data.drop(labels=["PX_VOLUME"], axis=1, inplace=True)
        market_data.rename(columns={'PX_OPEN': 'Open', 'PX_HIGH': 'High', 'PX_LOW': 'Low', 'PX_VOLUME': 'Volume',
                           'PX_LAST': HeaderFactory.Price}, inplace=True)
        market_data.dropna(inplace=True)
        return market_data