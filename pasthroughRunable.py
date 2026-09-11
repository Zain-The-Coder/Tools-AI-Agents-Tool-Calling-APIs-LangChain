from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnablePassthrough

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

code_prompt = ChatPromptTemplate.from_messages([
    {"system" , "You are a code generator"} , 
    {"human" , "{code}"}
])

explain_code_prompt = ChatPromptTemplate.from_messages([
    {"system" , "You are a helpful code explainer who expain code in simple way"} , 
    {"human" , "Explain this code : {topic}"}

])

parser = StrOutputParser()

seq1 = code_prompt | model | parser 

seq2 = RunnableParallel({
    "code" : RunnablePassthrough() ,
    "explanation" : explain_code_prompt | model | parser
})

chain = seq1 | seq2 


result = chain.invoke({"topic" : "Write bubble sorting algorithm in javascript"})

print(result['code'])
print(result['explanation'])