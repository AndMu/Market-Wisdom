import abc
import logging

import pandas as pd
from PortfolioBasic.stockstats import StockDataFrame

from PortfolioBasic.Definitions import HeaderFactory
from PortfolioBasic.Technical.Analysis import TechnicalPerformance
logger = logging.getLogger(__name__)


class Indicator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame)-> pd.DataFrame:
        pass

    @abc.abstractmethod
    def required_days(self) -> int:
        pass


class CombinedIndicator(Indicator):

    def required_days(self) -> int:
        return max(self.indicators, key=lambda x: x.required_days()).required_days()

    def __init__(self, indicators: list):
        self.indicators = indicators

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        for indicator in self.indicators:
            indicator_result = indicator.calculate(data)
            result = result.join(indicator_result)
        return result


class MomentumIndicator(Indicator):

    def __init__(self, days=5):
        self.days = days

    def required_days(self) -> int:
        return self.days + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[HeaderFactory.Price].copy()
        previous = data.shift(self.days)
        data = data / previous - 1
        result = pd.DataFrame(index=data.index, data=data.values, columns=['{}_{}'.format(HeaderFactory.MOM, self.days)])
        return result


class Williams(Indicator):
    def __init__(self):
        self.windows = 10

    def required_days(self) -> int:
        return self.windows

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        stock = StockDataFrame.retype(data.copy())
        wr_10 = stock['wr_10']
        result = pd.DataFrame(index=wr_10.index, data=wr_10.values, columns=['wr_10'])
        return result


class CommodityChannelIndex(Indicator):
    def __init__(self):
        self.windows = 14

    def required_days(self) -> int:
        return self.windows

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        stock = StockDataFrame.retype(data.copy())
        value = stock['cci']
        result = pd.DataFrame(index=value.index, data=value.values, columns=['cci'])
        return result


class TripleExponentialMovingAverage(Indicator):
    def __init__(self):
        self.windows = 12

    def required_days(self) -> int:
        return self.windows

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        stock = StockDataFrame.retype(data.copy())
        value = stock['trix']
        result = pd.DataFrame(index=value.index, data=value.values, columns=['trix'])
        return result


class AverageTrueRange(Indicator):
    def __init__(self):
        self.windows = 12

    def required_days(self) -> int:
        return self.windows

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        stock = StockDataFrame.retype(data.copy())
        value = stock['atr']
        result = pd.DataFrame(index=value.index, data=value.values, columns=['atr'])
        return result


class AverageDirectionalIndex(Indicator):
    def __init__(self):
        self.windows = 6

    def required_days(self) -> int:
        return self.windows

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        stock = StockDataFrame.retype(data.copy())
        value = stock['adx']
        result = pd.DataFrame(index=value.index, data=value.values, columns=['adx'])
        return result


class BollingerIndicator(Indicator):

    def __init__(self, windows=20):
        self.windows = windows

    def required_days(self) -> int:
        return self.windows

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[HeaderFactory.Price].copy()
        rm, rstd = TechnicalPerformance.compute_std(data, self.windows)
        result = (data - rm) / (2 * rstd)
        result = pd.DataFrame(index=result.index, data=result.values, columns=[HeaderFactory.Bollinger])
        return result


class RsiIndicator(Indicator):

    def required_days(self) -> int:
        return 15

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(data[HeaderFactory.Price])
        data = data.rename(index=str, columns={HeaderFactory.Price: "close"})
        stock = StockDataFrame.retype(data)
        rsi = stock['rsi_14']
        rsi /= 100
        data = rsi
        result = pd.DataFrame(index=data.index, data=data.values, columns=[HeaderFactory.RSI])
        return result


class MACDIndicator(Indicator):

    def __init__(self):
        self.n_fast = 12
        self.n_slow = 26
        self.signal_period = 9

    def required_days(self) -> int:
        return 26

    def calculate(self, data: pd.DataFrame, normalized=False) -> pd.DataFrame:
        fast = data[HeaderFactory.Price].ewm(adjust=True, min_periods=self.n_slow - 1, span=self.n_fast,
                                             ignore_na=False).mean()
        EMAfast = pd.Series(fast)

        slow = data[HeaderFactory.Price].ewm(adjust=True, min_periods=self.n_slow - 1, span=self.n_slow,
                                             ignore_na=False).mean()
        EMAslow = pd.Series(slow)
        result = EMAfast - EMAslow
        if normalized:
            result = result / EMAslow
        MACD = pd.Series(result, name=HeaderFactory.MACD)
        signal = MACD.ewm(adjust=True, min_periods=self.signal_period - 1, span=self.signal_period,
                          ignore_na=False).mean()
        MACDsign = pd.Series(signal, name=HeaderFactory.MACD_SIGNAL)
        MACDdiff = pd.Series(MACD - MACDsign, name=HeaderFactory.MACD_DIFF)
        data = pd.DataFrame(MACD).join(MACDsign)
        data = data.join(MACDdiff)
        return data

