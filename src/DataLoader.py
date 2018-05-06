import pandas as pd
import numpy as np
from functools import reduce

from PortfolioBasic.stockstats import StockDataFrame

from MarketData import QuandlMarketDataSource, MarketData
from PortfolioBasic.Definitions import HeaderFactory
from PortfolioBasic.Technical.Indicators import RsiIndicator, MomentumIndicator, MACDIndicator,  \
    CombinedIndicator, BollingerIndicator, Williams, CommodityChannelIndex, TripleExponentialMovingAverage, \
    AverageDirectionalIndex, AverageTrueRange


class DataLoader(object):

    def load_data(self, stock, days, sentiment_location=None, source=QuandlMarketDataSource(), full_articles=True,
                  from_date='2011-04-01', to_date='2015-04-01'):
        articles = None
        if sentiment_location is not None:
            articles = self.load_sentiment(sentiment_location, full_articles)
        price_df = source.get_stock_data(stock)
        # noinspection PyTypeChecker
        indicators = CombinedIndicator((
            MomentumIndicator(1),
            MomentumIndicator(5),
            BollingerIndicator(),
            MACDIndicator(),
            CommodityChannelIndex(),
            AverageDirectionalIndex(),
            TripleExponentialMovingAverage(),
            AverageTrueRange(),
            RsiIndicator(),
            Williams()
            ))
        ma = [
            50,
            100,
            200
        ]

        market = MarketData(stock, price_df, days=days, ma=ma, indicator=indicators, df_additional=articles)
        price_df = market.get_stock_data()
        return market.get_binary_data(price_df, days=days, from_date=from_date, to_date=to_date)

    def load_sentiment(self, location, full_articles=True):
        articles = pd.read_csv(location, na_values=["nan"])

        articles.Date = pd.to_datetime(articles.Date, format='%d/%m/%Y %H:%M:%S')
        articles.set_index('Date', inplace=True)
        articles.reindex()
        articles.sort_index(ascending=True, inplace=True)

        if full_articles:
            articles['Anger'] /= articles['TotalWords']
            articles['Anticipation'] /= articles['TotalWords']
            articles['Disgust'] /= articles['TotalWords']
            articles['Fear'] /= articles['TotalWords']
            articles['Joy'] /= articles['TotalWords']
            articles['Sadness'] /= articles['TotalWords']
            articles['Surprise'] /= articles['TotalWords']
            articles['Trust'] /= articles['TotalWords']
            articles.drop(columns=['Original', 'Original', 'TotalWords', 'TotalSentimentWords', 'Id'], inplace=True)
        else:
            articles.drop(columns=['Original', 'Original', 'TotalWords', 'TotalSentimentWords', 'Id', 'Anger',
                                   'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'], inplace=True)
        articles['Calculated'] = ((articles['Calculated'] - 1) / 2) - 1
        articles = articles.groupby(pd.TimeGrouper('D')).mean()
        articles.to_csv('articles.csv')
        return articles

    def load(self, days: int = 5, include_articles: bool=True):

        market_data = self.load_bloomber()
        market_data.sort_index(inplace=True)
        prices = pd.DataFrame(market_data[HeaderFactory.Price])

        if include_articles:
            articles = self.load_sentiment()
            prices = prices.join(articles)

        rsi_indicator = RsiIndicator()
        rsi = rsi_indicator.calculate(market_data)

        momentum_indicator = MomentumIndicator(1)
        momentum = momentum_indicator.calculate(market_data)

        momentum_indicator_5 = MomentumIndicator(5)
        momentum_5 = momentum_indicator_5.calculate(market_data)
        momentum_5 = momentum_5.rename(columns={"MOM": "MOM_5"})

        stock = StockDataFrame.retype(market_data.copy())
        wr_10 = stock['wr_10']
        wr_10 = pd.DataFrame(index=wr_10.index, data=wr_10.values, columns=['wr_10'])

        macd_indicator = MACDIndicator()
        macd = macd_indicator.calculate(market_data)
        macd_diff = pd.DataFrame(macd[HeaderFactory.MACD_DIFF])
        macd_x = pd.DataFrame(macd[HeaderFactory.MACD])
        macd_sig = pd.DataFrame(macd[HeaderFactory.MACD_SIGNAL])

        y_data = pd.DataFrame(market_data[HeaderFactory.Price])
        y_data['y'] = np.where(y_data.shift(-days) < y_data, 0, 1)
        y_data.drop(columns=[HeaderFactory.Price], inplace=True)

        data_frames = [
            prices,
            macd_diff,
            macd_x,
            macd_sig,
            momentum,
            momentum_5,
            rsi,
            wr_10,
            y_data]
        df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'),
                           data_frames)
        df_merged.fillna(value=0, inplace=True)
        df_merged = df_merged.ix['2011-04-01': '2015-04-01'].copy()
        df_merged.to_csv('result.csv')
        x_data = df_merged.iloc[1:-days, 1:-1].values
        y_data = df_merged.iloc[1:-days, -1].values
        return x_data, y_data