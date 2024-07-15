import os
import shutil
import nltk
import ollama
import numpy as np
import faiss
from langchain.evaluation import load_evaluator
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
import streamlit as sl

DATA_PATH = "data/books"

# Load the PDF
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    docs = loader.load()
    return docs

# Split into chunks
def split_text(documents: list[Document]):  
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split{len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

CHROMA_PATH = "chroma"
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved{len(chunks)} chunks to {CHROMA_PATH}.")

# Instantiate the embedding model
embedder = OllamaEmbeddings()

# Create the vector store and fill it with embeddings
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define llm
llm = Ollama(model="mistral")

# Define the prompt
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

text = "Your long text goes here. It will be split into smaller chunks while maintaining sentence structure. This helps in ensuring the coherence of each chunk."


    #function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        DB_FAISS_PATH = '../vectorstore'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        llm = Ollama(model="llama3")
        return llm

#creating prompt template using langchain
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
        sl.header("welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by entering your queries ")
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        query=sl.text_input('Enter some text')
        
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                similar_embeddings=knowledgeBase.similarity_search(query)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                
                #creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                response=rag_chain.invoke(query)
                sl.write(response)