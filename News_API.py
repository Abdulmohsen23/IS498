import http.client
import json

def fetch_stock_news(api_token, symbols, published_after, filename):
    conn = http.client.HTTPSConnection('api.marketaux.com')

    symbols_param = ','.join(symbols)
    url = f"/v1/news/all?countries=sa&filter_entities=true&limit=10&symbols={symbols_param}&published_after={published_after}&api_token={api_token}"
    conn.request('GET', url)

    res = conn.getresponse()
    if res.status != 200:
        print(f"Error: {res.status} - {res.reason}")
        return

    data = res.read().decode('utf-8')
    
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(json.loads(data), file, indent=4)

# Replace 'YOUR_API_TOKEN' with your actual API token
api_token = 'exUBYXyWyKyzEmeuLf7Sg1GpXhEmJBtX4bETpLcb'
symbols = ['8200.SR', '1150.SR', '2350.SR', '4040.SR', '4170.SR', '4015.SR', '4270.SR', '7030.SR', '4070.SR', '6010.SR', '4110.SR', '2230.SR', '4061.SR', '4300.SR', '4190.SR', '2380.SR', '1040.SR', '4180.SR', '4130.SR', '5110.SR', '4331.SR', '7201.SR']
published_after = '2011-04-04T00:00'  # Specify the desired date and time
output_filename = 'stock_news.json'  # Specify the filename to write the data to

fetch_stock_news(api_token, symbols, published_after, output_filename)
print("Data written to", output_filename)
