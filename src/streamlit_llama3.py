#import Essential dependencies
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os
import shutil
from typing import List
from langchain.schema import Document

# Specify the correct path to your documents directory
DATA_PATH = '../data'
DB_FAISS_PATH = './vectorstore'
# CHROMA_PATH = "./chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def split_text(docs, max_length=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=chunk_overlap
        )
    chunks = []
    return chunks

def create_knowledgeBase():
    docs = load_documents()
    chunks = split_text(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

def load_knowledgeBase():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    llm = Ollama(model="llama3")
    return llm

def load_prompt():
    prompt = """
    You need to answer the question in the sentence as same as in the PDF content.
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    If the answer is not in the PDF, answer "Sorry, I'm not sure how to respond to this"
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
    # Creates header for streamlit app and writes to it
    st.header("welcome to the 📝PDF bot")
    st.write("🤖 You can chat by entering your queries ")
        
    # Creates and loads all of components for RAG system
    create_knowledgeBase()
    knowledgeBase=load_knowledgeBase()
    llm=load_llm()
    prompt=load_prompt()
        
    # Creates text box for user to query data
    query=st.text_input('Enter some text')
        
    if(query):
            # Gets most similar vectors from knowledge base to user query and turns into actual documents
            similar_embeddings=knowledgeBase.similarity_search(query)
            similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                
            # Defines chain together query, documents, prompt, and LLM to form process for generating response
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Calls chain and writes response to streamlit
            response=rag_chain.invoke(query)
            st.write(response)
