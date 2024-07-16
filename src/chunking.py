import os
import shutil
from typing import List
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

DATA_PATH = "random machine learing pdf.pdf"
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    docs = loader.load()
    return docs

def split_text(documents: List[str]):
    return documents  # Placeholder function, replace with actual text splitting logic

def save_to_chroma(chunks: List[str]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    document_chunks = [Document(page_content=chunk) for chunk in chunks]
    db = Chroma.from_documents(document_chunks, embeddings, persist_directory=CHROMA_PATH)
    return db

def load_knowledge_base():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    DB_FAISS_PATH = '../vectorstore'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    llm = Ollama(model="llama3")
    return llm

def load_prompt():
    prompt = """
    You need to answer the question in the sentence as same as in the pdf content. 
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    If the answer is not in the pdf, answer "Sorry, I'm not sure how to respond to this."
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

if __name__ == '__main__':
    st.header("Welcome to the 📝PDF bot")
    st.write("🤖 You can chat by entering your queries")
    
    # Load necessary components
    knowledge_base = load_knowledge_base()
    llm = load_llm()
    prompt = load_prompt()
    
    query = st.text_input('Enter your query')
    
    if query:
        # Perform operations based on user query
        chunks = split_text([query])  # Placeholder function, replace with actual text splitting logic
        db = save_to_chroma(chunks)
        
        # Example usage of RAG (Retrieval-Augmented Generation) workflow
        retriever = knowledge_base.as_retriever()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        
        st.write(response)
