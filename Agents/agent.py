from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, ToolMessage
from agentTools import get_updated_news, get_weather
from rich import print
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="conversational",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

tools = {
    "get_updated_news": get_updated_news,
    "get_weather": get_weather
}

# binding tool
model_with_tool = model.bind_tools([get_weather, get_updated_news])

query = input("You : ")

chat_history = []
chat_history.append(HumanMessage(content=query))

tool_response = model_with_tool.invoke(chat_history)
chat_history.append(tool_response)      

if tool_response.tool_calls:
    for tool in tool_response.tool_calls:
        tool_name = tool["name"]
        tool_args = tool['args']
        tool_id = tool['id']

        selected_tool = tools.get(tool_name)
        tool_output = selected_tool.invoke(tool_args)

        chat_history.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

    final_response = model_with_tool.invoke(chat_history) 
    print(final_response.content)

else:
    print(tool_response.content)          



print(f"Chat History : {chat_history}")