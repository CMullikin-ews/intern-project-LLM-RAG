import os
import logging
import streamlit as st
import streamlit_llama3
import random
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Initialize Ollama LLM
llm = Ollama(model='llama3')
class RAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        documents = self.retriever.get_relevant_documents(query)
        # Process documents with the LLM
        response = self.llm.generate(query, documents)
        return response
    
retriever = BM25Retriever()
pipeline = RAGPipeline(llm=llm, retriever=retriever)

DATA_PATH = '../../cyber_data'
DB_FAISS_PATH = '../vectorstore'

# Initialize document store
document_store = InMemoryDocstore()

# Add documents to the store (this is an example; replace with your documents)
documents = [
    {"content": "../cyber_data", "metadata": {"source": "Wikipedia"}},
    # Add more documents as needed
]
document_store.write_documents(documents)

# Initialize retriever
retriever = BM25Retriever(document_store=document_store)

# Define a prompt template for RAG
prompt_template = PromptTemplate(
    template="Context: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

# Create an LLM chain for RAG
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(retriever=retriever, generator=llm_chain)

# Streamlit user interface
st.title("Advanced RAG System with LangChain and Ollama")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    # Retrieve and generate answer using RAG pipeline
    response = rag_pipeline.run({"question": question})
    st.write("Answer:", response)