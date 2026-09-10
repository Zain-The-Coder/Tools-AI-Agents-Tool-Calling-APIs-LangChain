from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)
# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="conversational",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

#       ALL STEPS ARE REPLACED BY RUNABLES

# formatted_prompt = prompt.format_messages(topic = "MERN Stack Development")

# response = model.invoke(formatted_prompt)

# final_output = parser.parse(response.content)

chain = prompt  | model | parser
output = chain.invoke({"topic" : "MERN Stack"})
print(output)