from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain.tools import tool
from rich import print
import os

load_dotenv()

#1 CREATING A TOOL
@tool
def text_length_measure (text : str) -> int :
    """Returns the number of characters in a given text"""
    return len(text)

#2 - get LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="conversational",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

#3 - Binding Tools
model_lengthMeasurementTool = model.bind_tools([text_length_measure])

#4 - We Give Tool to LLM (LLM will Decides Tool)
result = model_lengthMeasurementTool.invoke("Return the number of character in a given text : Hello how are you")

#5 - Tool execution
if result.tool_calls:
    tool_call = result.tool_calls[0]

    args = tool_call["args"]

    tool_result = text_length_measure.invoke(args)

    final_result = model.invoke(
        f"The length of the text is {tool_result}"
    )

    print(final_result)

else:
    print("NO Tool Calls Found")
