from sklearn import pipeline
import streamlit as sl
import streamlit_llama3
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
from typing import List
import os
import logging
import random

# Define data paths
DATA_PATH = '../../cyber_data'
DB_FAISS_PATH = '../vectorstore'

# File loaders
loaders = {
    '.php': UnstructuredFileLoader,
    '.cs': UnstructuredFileLoader,
    '': UnstructuredFileLoader,
    '.c': UnstructuredFileLoader,
    '.html': UnstructuredHTMLLoader,
    '.md': UnstructuredMarkdownLoader,
    '.txt': TextLoader,
    '.ps1': UnstructuredFileLoader,
    '.delphi': UnstructuredFileLoader,
    '.asm': UnstructuredFileLoader,
    '.TXT': TextLoader
}

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set environment variables for Ollama
os.environ["OLLAMA_HOST"] = "localhost:8501"

def setup_ollama():
    try:
        os.system("curl -fsSL https://ollama.com/install.sh | sh")
        os.system("sudo pkill ollama")
        os.system("ollama serve")
        os.system("ollama pull mxbai-embed-large")
        os.system("ollama pull jimscard/whiterabbit-neo")
        logging.info("Ollama setup completed successfully.")
    except Exception as e:
        logging.error(f"Error setting up Ollama: {e}")

def get_file_types(directory):
    file_types = set()
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            _, ext = os.path.splitext(filename)
            file_types.add(ext)
    return file_types

def create_directory_loader(file_type, directory_path):
    """
    Creates and returns a DirectoryLoader using the loader specific to the file type provided
    
    Args:
        file_type (str): Type of file to make loader for
        directory_path (str): Path to directory

    Returns:
        DirectoryLoader: loader for the files in the directory provided
    """
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders.get(file_type, UnstructuredFileLoader)
    )

def load_documents():
    file_types = get_file_types(DATA_PATH)
    documents = []
    for file_type in file_types:
        if file_type.strip():
            loader = create_directory_loader(file_type, DATA_PATH)
            docs = loader.load()
            chunks = split_text(docs)
            if chunks:
                documents.extend(chunks)
    return documents

def split_text(docs, max_length=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def create_knowledgeBase():
    docs = load_documents()
    chunks = split_text(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

def load_knowledgeBase():
    docs = load_documents()
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(DB_FAISS_PATH)
    return vector_store

def load_llm():
    return Ollama(model="llama3")

def load_prompt():
    prompt_text = """
    You are an assistant for helping software developers to detect and neutralize viruses.
    Make sure to clearly define any necessary terms and go through the steps to use any application or software.
    Only use the data provided to you.
    Cite the sources used in constructing the response.
    If the answer is not in the data provided, answer "Sorry, I'm not sure how to respond to this"
    """
    return ChatPromptTemplate.from_template(prompt_text)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(query: str) -> List[str]:
    responses = [
        "Sure, I can help with that!",
        "Let me find that information for you.",
        "Here is what I found.",
        "This is the information you requested."
    ]
    return random.choice(responses)

def get_relevant_url(query: str) -> List[str]:
    urls = [
        "https://example.com/info1",
        "https://example.com/info2",
        "https://example.com/info3",
        "https://example.com/info4"
    ]
    return random.choice(urls)

def respond_with_url(query: str) -> List[str]:
    retrieved_docs = retriever.retrieve(query)
    sources = [doc.metadata['source'] for doc in retrieved_docs]
    response = pipeline.generate(query)
    citation_text = "Sources: " + ", ".join(sources)
    return f"{response}\n\n{citation_text}"

if __name__ == '__main__':
    setup_ollama()
    sl.header("Welcome to the 📝PDF bot")
    sl.write("🤖 You can chat by entering your queries")
    
    try:
        knowledge_base = load_knowledgeBase()
        llm = load_llm()
        prompt = load_prompt()
        logging.info("Components loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading components: {e}")
        sl.write("An error occurred while loading the components. Please check the logs.")

    query = sl.text_input('Enter some text')
    
    if query:
        try:
            similar_embeddings = streamlit_llama3.knowledge_base.similarity_search(query)
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            response = rag_chain.invoke(query)
            sl.write(response)
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            sl.write("An error occurred while processing your query. Please check the logs.")
