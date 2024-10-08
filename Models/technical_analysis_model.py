import numpy as np
import pandas as pd

class TechnicalAnalysisModel:
    def __init__(self, rsi_window=100, bb_window=30, bb_std=2):
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
    
    def train(self, data):
        # No training required for this model
        pass
    
    def calculate_rsi(self, data):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, data):
        rolling_mean = data['Close'].rolling(window=self.bb_window).mean()
        rolling_std = data['Close'].rolling(window=self.bb_window).std()
        upper_band = rolling_mean + (rolling_std * self.bb_std)
        lower_band = rolling_mean - (rolling_std * self.bb_std)
        return upper_band, rolling_mean, lower_band
    
    def predict(self, data):
        # Calculate RSI
        data['RSI'] = self.calculate_rsi(data)
        
        # Calculate Bollinger Bands
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self.calculate_bollinger_bands(data)
        
        # Generate buy/sell signals based on technical indicators
        data['Signal'] = 0
        data.loc[(data['Close'] > data['BB_Upper']) & (data['RSI'] > 70), 'Signal'] = -1
        data.loc[(data['Close'] < data['BB_Lower']) & (data['RSI'] < 30), 'Signal'] = 1
        
        return data['Signal']

# import pandas as pd
# import numpy as np

# class TechnicalAnalysisModel:
#     def __init__(self, rsi_window=100, bb_window=30, bb_std=3, macd_fast=21, macd_slow=34, macd_signal=14, stoch_k=21, stoch_d=7):
#         self.rsi_window = rsi_window
#         self.bb_window = bb_window
#         self.bb_std = bb_std
#         self.macd_fast = macd_fast
#         self.macd_slow = macd_slow
#         self.macd_signal = macd_signal
#         self.stoch_k = stoch_k
#         self.stoch_d = stoch_d

#     def train(self, data):
#         # No training required for this model
#         pass

#     def calculate_rsi(self, data):
#         delta = data['Close'].diff()
#         gain = delta.where(delta > 0, 0)
#         loss = -delta.where(delta < 0, 0)
#         avg_gain = gain.rolling(window=self.rsi_window).mean()
#         avg_loss = loss.rolling(window=self.rsi_window).mean()
#         rs = avg_gain / avg_loss
#         rsi = 100 - (100 / (1 + rs))
#         return rsi

#     def calculate_bollinger_bands(self, data):
#         rolling_mean = data['Close'].rolling(window=self.bb_window).mean()
#         rolling_std = data['Close'].rolling(window=self.bb_window).std()
#         upper_band = rolling_mean + (rolling_std * self.bb_std)
#         lower_band = rolling_mean - (rolling_std * self.bb_std)
#         return upper_band, rolling_mean, lower_band

#     def calculate_macd(self, data):
#         exp1 = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
#         exp2 = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
#         macd = exp1 - exp2
#         signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
#         hist = macd - signal
#         return macd, signal, hist

#     def calculate_stochastic(self, data):
#         low_rolling = data['Low'].rolling(window=self.stoch_k).min()
#         high_rolling = data['High'].rolling(window=self.stoch_k).max()
#         k = 100 * (data['Close'] - low_rolling) / (high_rolling - low_rolling)
#         d = k.rolling(window=self.stoch_d).mean()
#         return k, d

#     def predict(self, data):
#         data.loc[:, 'RSI'] = self.calculate_rsi(data)
#         data.loc[:, ['BB_Upper', 'BB_Middle', 'BB_Lower']] = self.calculate_bollinger_bands(data)
#         data.loc[:, ['MACD', 'MACD_Signal', 'MACD_Hist']] = self.calculate_macd(data)
#         data.loc[:, ['SlowK', 'SlowD']] = self.calculate_stochastic(data)
#         data.loc[:, 'Signal'] = 0

#         # Generate buy/sell signals based on technical indicators
#         data.loc[(data['Close'] > data['BB_Upper']) & (data['RSI'] > 80) & (data['MACD'] > data['MACD_Signal']) & (data['SlowK'] > 90), 'Signal'] = -1
#         data.loc[(data['Close'] < data['BB_Lower']) & (data['RSI'] < 20) & (data['MACD'] < data['MACD_Signal']) & (data['SlowD'] < 10), 'Signal'] = 1

#         return data['Signal']
