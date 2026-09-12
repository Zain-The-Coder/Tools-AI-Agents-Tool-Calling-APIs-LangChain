from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from rich import print
import requests
import os

load_dotenv()

@tool
def get_current_weather() -> str:
    """Get the current weather of Karachi."""

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast?latitude=24.86&longitude=67.01&current=temperature_2m,wind_speed_10m"
    )

    result = response.json()

    temperature = result["current"]["temperature_2m"]
    wind_speed = result["current"]["wind_speed_10m"]
    time = result["current"]["time"]

    now = time.split("T")

    current_time = {
        "date": now[0],
        "time": now[1],
        "temperature": temperature,
        "wind_speed": wind_speed
    }

    return f"""At {current_time["time"]} on {current_time["date"]}
The temperature is {current_time["temperature"]}°C and wind speed is {current_time["wind_speed"]} km/h"""


#2 - get LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

#binding tool

weather_model = model.bind_tools([get_current_weather])

result = weather_model.invoke("Get the current weather of Karachi.")

if result.tool_calls :
    tool_call = result.tool_calls[0]

    args = tool_call['args']

    tool_result = get_current_weather.invoke(args)

    final_result = model.invoke(
        f"the current weather information is {tool_result}"
    )

    print(final_result.content)
else:
    print("NO Tool Calls Found")
