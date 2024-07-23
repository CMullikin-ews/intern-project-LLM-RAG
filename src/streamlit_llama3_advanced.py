import streamlit as sl
import streamlit_llama3
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
import os
import logging
import textract
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Define data paths
DATA_PATH = '../../cyber_data'
DB_FAISS_PATH = '../vectorstore'

# File loaders
loaders = {
    '.php': UnstructuredFileLoader,
    '.cs': UnstructuredFileLoader,
    '.': UnstructuredFileLoader,
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
logger = logging.getLogger(__name__)

def setup_ollama():
    try:
        os.system("ollama serve")
        os.system("ollama pull mxbai-embed-large")
        os.system("ollama pull jimscard/whiterabbit-neo")
        logging.info("Ollama setup completed successfully.")
    except Exception as e:
        logging.error(f"Error setting up Ollama: {e}")

def get_file_types(directory):
    file_types = set()
    for root, _, files in os.walk(directory):
        for file in files:
            file_type = os.path.splitext(file)[1]
            file_types.add(file_type)
    return file_types

def extract_text(file_path):
    try:
        text = textract.process(file_path).decode('utf-8')
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None

def extract_metadata(text):
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word.lower() not in stopwords.words('english')]
    word_counts = Counter(filtered_words)
    metadata = {
        'total_words': len(word_tokens),
        'unique_words': len(set(filtered_words)),
        'word_counts': word_counts
    }
    return metadata

def process_document(file_path):
    text = extract_text(file_path)
    if text:
        metadata = extract_metadata(text)
        print(metadata)
        return metadata
    return {}

def load_and_process_documents(directory_path):
    loader = DirectoryLoader(directory_path, loader_cls=TextLoader)
    documents = loader.load()
    for doc in documents:
        file_path = doc['source']
        metadata = process_document(file_path)
        # Store or use the metadata as needed

if __name__ == '__main__':
    setup_ollama()
    sl.header("Welcome to the 📝Computer Virus Copilot")
    sl.write("🤖 You can chat by entering your queries")

    try:
        knowledge_base = streamlit_llama3.load_knowledgeBase()
        llm = streamlit_llama3.load_llm()
        prompt = streamlit_llama3.load_prompt()
        logging.info("Components loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading components: {e}")
        sl.write("An error occurred while loading the components. Please check the logs.")

    query = sl.text_input('Enter some text')
    
    if query:
        try:
            similar_embeddings = knowledge_base.similarity_search(query)
            similar_embeddings = FAISS.from_documents(
                documents=similar_embeddings,
                embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
            )
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


