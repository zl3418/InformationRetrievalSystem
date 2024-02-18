import requests

def search_google(query, google_api_key, google_engine_id):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_api_key,
        "cx": google_engine_id,
        "q": query,
        "num": 10
    }
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    return results
