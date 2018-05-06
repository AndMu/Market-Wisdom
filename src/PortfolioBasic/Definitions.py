from pandas import DataFrame


class HeaderFactory(object):
    Columns = ["Open", "High", "Low", "Close", "Volume"]
    Index = 'Index'
    Price = 'Close'
    Order = 'Order'
    Shares = 'Shares'
    BUY = 'BUY'
    EXIT = 'EXIT'
    SELL = 'SELL'
    Bollinger = 'Bollinger'
    Low = "Low"
    High = "High"
    StopLow = "StopLow"
    StopHigh = "StopHigh"
    ADR = "ADR"
    MACD = 'MACD'
    MACHINE = 'MACHINE'
    RSI = 'RSI'
    MOM = 'MOM'
    MACD_HIST= 'MACD_HIST'
    MACD_SIGNAL = 'MACD_SIGNAL'
    MACD_DIFF = 'MACD_DIFF'
    SMA = 'SMA'
    RSTD = 'RSTD'
    BB_UPPER_BAND = 'UPPER_BAND'
    BB_LOWER_BAND = 'LOWER_BAND'

    @staticmethod
    def get_name(symbol: str, type: str):
        return symbol + '_' + type


class Utilities:
    @staticmethod
    def fill_missing_values(df_data: DataFrame):
        """Fill missing values in data frame, in place."""
        df_data.fillna(method='ffill', inplace=True)
        df_data.fillna(method='bfill', inplace=True)