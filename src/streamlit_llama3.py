#import Essential dependencies
import streamlit as st
import os
import shutil
from typing import List
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS, Chroma
from langchain.runnables import RunnablePassthrough
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        st.header("Welcome to the 📝PDF bot")
        st.write("🤖 You can chat by entering your queries")   
        knowledgeBase = load_knowledgeBase()
        llm = load_llm()
        prompt = load_prompt()
        query = st.text_input('Enter some text')
    
        if query:
            similar_embeddings = knowledgeBase.similarity_search(query)
            documents = [Document(page_content=doc.page_content) for doc in similar_embeddings]
            formatted_docs = format_docs(documents)
        
            retriever = knowledgeBase.as_retriever()
            rag_chain = (
                {"context": formatted_docs, "question": RunnablePassthrough(query)}
                | prompt
                | llm
                | StrOutputParser()
            )
        
            response = rag_chain.invoke(query)
            st.write(response)
