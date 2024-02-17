import requests

API_KEY = "AIzaSyA-qNsQ7hExo8A0ZYVz7Vwq-5jIetGoF8o"
CSE_ID = "c5cfab8a2ca6a4074"

def search_google(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": 10
    }
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    return results
