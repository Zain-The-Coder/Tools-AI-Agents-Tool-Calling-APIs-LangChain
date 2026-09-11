from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
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
parser = StrOutputParser()

search_tool = TavilySearchResults(max_result = 3)

prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant
    summarize the following news into clear bullets points and also write the name of author and also mention the date of news
    {news}
""")

chain = prompt | model | parser

news_result = search_tool.run("Latest AI news of 2026")

result = chain.invoke({"news" : news_result})

print(result)