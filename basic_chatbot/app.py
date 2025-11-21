from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## environment variables call

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## Langsmith Tracking Setup

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## creating chatbot

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide response to the user queries based on the context provided."),
    ("user", "Question: {question}"),
])

## streamlit framework

st.title("Basic Chatbot with LangChain and OpenAI")
input_text = st.text_input("Search the topic you want:") 

## open AI LLM call
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
output_parser = StrOutputParser()

## chain
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))