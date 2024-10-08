# from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# class ExponentialSmoothingModel:
#     def __init__(self, smoothing_level=0.2):
#         self.smoothing_level = smoothing_level
#         self.model = None
    
#     def train(self, data):
#         self.model = SimpleExpSmoothing(data['Close']).fit(smoothing_level=self.smoothing_level, optimized=False)
    
#     def predict(self, data):
#         data['ES_Forecast'] = self.model.forecast(len(data))
        
#         # Generate buy/sell signals based on exponential smoothing forecast
#         data['Signal'] = 0
#         data.loc[data['Close'] > data['ES_Forecast'], 'Signal'] = 1
#         data.loc[data['Close'] < data['ES_Forecast'], 'Signal'] = -1
        
#         return data['Signal']

from statsmodels.tsa.arima.model import ARIMA

class ExponentialSmoothingModel:
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.model = None

    def train(self, data):
        self.model = ARIMA(data['Close'].values, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, data):
        data.loc[:, 'ES_Forecast'] = self.model_fit.forecast(steps=len(data))
        data.loc[:, 'Signal'] = 0
        data.loc[data['Close'] > data['ES_Forecast'], 'Signal'] = 1
        data.loc[data['Close'] < data['ES_Forecast'], 'Signal'] = -1
        return data['Signal']