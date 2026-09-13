from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from rich import print
import os
import requests

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

tools = {
    "get_current_weather" : get_current_weather
}

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

model_with_tool = model.bind_tools([get_current_weather])
chat_history = []

query = HumanMessage("Get the current weather of Karachi.")
chat_history.append(query)

result = model_with_tool.invoke(chat_history)

chat_history.append(result)

if result.tool_calls : 
    tool_name = result.tool_calls[0]['name']
    tool_result = tools[tool_name].invoke(result.tool_calls[0])
    chat_history.append(tool_result)

    final_result = model_with_tool.invoke(chat_history)
    print(final_result.content)
else : 
    print("No Tool Calls Found !")