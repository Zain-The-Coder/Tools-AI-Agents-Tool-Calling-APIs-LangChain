from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel , RunnableLambda
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

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in just 1 line"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

#for same question in parallel
    
parallelChain = RunnableParallel({
    "short" : short_prompt | model | parser ,
    "detailed" : detailed_prompt | model | parser
})

result = parallelChain.invoke({"topic" : "MERN Stack"})
print(result['short'])
print(result["detailed"])

#for different question


parallelChain2 = RunnableParallel({
    "short" : RunnableLambda(lambda x : x['short']) | short_prompt | model | parser ,
    "detailed" : RunnableLambda(lambda x : x['detailed']) | detailed_prompt | model | parser
})

result = parallelChain2.invoke({
    "short" : {"topic" : "Programming"} ,
    "detailed" : {"topic" : "deployment"} 
})

print(result["short"])
print(result["detailed"])