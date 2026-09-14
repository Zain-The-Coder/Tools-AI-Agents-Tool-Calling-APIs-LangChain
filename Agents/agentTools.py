from langchain_core.tools import tool
from rich import print
from dotenv import load_dotenv
from tavily import TavilyClient
import os
import requests

load_dotenv()

@tool
def get_weather (city : str) -> str:
    """Get current weather of city"""

    API_KEY= os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    status = data.get("cod")

    if status != 200 :
        return f"Error : {data.get('message' , "could not fetch weather")}"

    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f'Weather in {city} : {desc} , {temp}°C'


tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@tool
def get_updated_news (city : str) -> str:
    """Get latest news about the city"""

    response = tavily_client.search(
        query=f"Latest news in {city}" ,
        max_results=1
    )

    results = response.get("results" , [])

    if not results :
        return f'No updated news found in {city}'

    news_list = []

    for r in results :
        title = r.get("title" , "No title")
        url = r.get('url' , '')
        snippet = r.get('content' , '')
        
        news_list.append(
        f'- {title}\n 📌{url}\n 📝{snippet[:400]}...'   
    )
        
    return f'Latest news in {city} : \n \n' + "\n\n".join(news_list)
