import http.client
import urllib.parse
import json

conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': 'ac0b27e8a70aeab1d594428b45e27515',
    'countries': 'us',
    'categories': 'business',
    'sort': 'published_desc',
    'date': '2020-01-01,2023-03-31',  # Retrieve news from January 1, 2023, to March 31, 2023
    'limit': 10,
})

try:
    conn.request('GET', '/v1/news?{}'.format(params))
    res = conn.getresponse()
    data = res.read()

    if res.status == 200:
        parsed_data = json.loads(data.decode('utf-8'))

        if 'data' in parsed_data:
            articles = parsed_data['data']
            if len(articles) > 0:
                for article in articles:
                    print("Title:", article['title'])
                    print("Description:", article['description'])
                    print("URL:", article['url'])
                    print("Published At:", article['published_at'])
                    print("----------------------")
            else:
                print("No news articles found within the specified date range.")
        else:
            print("Unexpected API response format. 'data' key not found.")
    else:
        print(f"API request failed with status code: {res.status}")
        print("Response:", data.decode('utf-8'))

except Exception as e:
    print("An error occurred:", str(e))