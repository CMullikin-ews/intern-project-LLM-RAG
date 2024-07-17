#import Essential dependencies
import streamlit as sl
import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

#Specify the correct path to your documents directory
DATA_PATH = '../data'
DB_FAISS_PATH = './vectorstore'
CHROMA_PATH = "./chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def split_text(text, max_length=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

def create_knowledgeBase():
    docs = load_documents()
    chunks = split_text(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

def save_to_chroma(chunks: List[str]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    document_chunks = [Document(page_content=chunk) for chunk in chunks]
    db = Chroma.from_documents(document_chunks, embeddings, persist_directory=CHROMA_PATH)
    return db

def load_knowledgeBase():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db
        
def load_llm():
    llm = Ollama(model="llama3")
    return llm

def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    if the answer is not in the pdf answer "Sorry, I'm not sure how to respond to this"
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
    sl.header("Welcome to the 📝PDF bot")
    sl.write("🤖 You can chat by entering your queries ")
    create_knowledgeBase()
    knowledgeBase = load_knowledgeBase()
    llm = load_llm()
    prompt = load_prompt()
    query = sl.text_input('Enter some text')
        
    if query:
            similar_embeddings = knowledgeBase.similarity_search(query)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
                
            response = rag_chain.invoke(query)
            sl.write(response)
                