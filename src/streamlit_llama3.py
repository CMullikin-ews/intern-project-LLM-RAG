# Packages used in RAG system
import streamlit as sl
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

DATA_PATH = '../data'
DB_FAISS_PATH = '../vectorstore'
# CHROMA_PATH = "chroma"

def load_documents():
        """
        Loads in files from ../data directory and returns them

        Returns:
                List[Document]: Array of documents
        """
        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        return docs

def split_text(docs, chunk_size=512, chunk_overlap=50):
        """
        Splits the given text into chunks of a specified maximum length using RecursiveCharacterTextSplitter.
        
        Parameters:
                text (str): The input text to be split.
                max_length (int): The maximum length of each chunk.
                chunk_overlap (int): The number of characters to overlap between chunks.
                
        Returns:
                List[str]: A list of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return chunks

def create_knowledgeBase():
        """
        Loads in documents, splits into chunks, and vectorizes chunks and stores vectors under FAISS vector store
        """
        docs = load_documents()
        documents = split_text(docs)
        embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectorstore.save_local(DB_FAISS_PATH)

# def save_to_chroma(chunks: List[str]):
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

#     embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
#     document_chunks = [Document(page_content=chunk) for chunk in chunks]
#     db = Chroma.from_documents(document_chunks, embeddings, persist_directory=CHROMA_PATH)
#     return db

def load_knowledgeBase():
        """
        Loads and returns vector store

        Returns:
            FAISS: vector store
        """
        embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db
        
def load_llm():
        """
        Creates and returns Llama3 model

        Returns:
            Llama3: LLM
        """
        llm = Ollama(model="llama3")
        return llm

#creating prompt template using langchain
def load_prompt():
        """
        Creates and returns prompt for LLM query that specifies how response sounds and structure of response

        Returns:
            ChatPromptTemplate: Prompt for LLM
        """
        prompt = """
        Imagine you are a knowledgeable and engaging teacher. Your task is to explain topic to a student in a clear and educational manner.
        Please include obscenities in the response.
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        If the answer is not in the data provided answer "Sorry, I'm not sure how to respond to this"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

def format_docs(docs):
        """
        Joins documents retrieved from vector store in one line format to make it easier for LLM to parse
        
        Args:
            docs (Document): Documents from vector stores

        Returns:
            String: documents in one line
        """
        return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
        # Creates header for streamlit app and writes to it
        sl.header("Welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by entering your queries")
        
        # Creates and loads all of components for RAG system
        # create_knowledgeBase()
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        
        # Creates text box for user to query data
        query=sl.text_input('Enter some text')
        
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
                sl.write(response)
                