import requests
from datetime import datetime, timedelta

def get_eod_data(symbol, date):
    params = {
        'access_key': '8fdf06826fee177418db6b1320229887',
        'symbols': symbol,
        'date_from': date,
        'date_to': date
    }

    api_result = requests.get('https://api.marketstack.com/v1/eod', params)
    api_response = api_result.json()

    if 'error' in api_response:
        error_message = api_response['error']['message']
        print(f"Error: {error_message}")
    elif 'data' in api_response:
        if api_response['data']:
            for stock_data in api_response['data']:
                print("Ticker:", stock_data['symbol'])
                print("Date:", stock_data['date'])
                print("Open:", stock_data['open'])
                print("High:", stock_data['high'])
                print("Low:", stock_data['low'])
                print("Close:", stock_data['close'])
                print("Volume:", stock_data['volume'])
                print("Exchange:", stock_data['exchange'])
                print("----------------------")
        else:
            print(f"No end-of-day data found for the symbol {symbol} on {date}")
    else:
        print("Unexpected API response structure.")

# Example usage
symbols = [
"4321.XSAU",
"2350.XSAU",
"4030.XSAU",
"8210.XSAU",
"2310.XSAU",
"1020.XSAU",
"4100.XSAU",
"2002.XSAU",
"1030.XSAU",
"4300.XSAU",
"2330.XSAU",
"2250.XSAU",
"3030.XSAU",
"4002.XSAU",
"8010.XSAU",
"3050.XSAU",
"2060.XSAU",
"4220.XSAU",
"6004.XSAU"
]

today = datetime.today().strftime('%Y-%m-%d')
previous_day = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

for symbol in symbols:
    get_eod_data(symbol, previous_day)