import streamlit as sl
import streamlit_llama3
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
import os
import logging
import textract
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
        os.system("curl -fsSL https://ollama.com/install.sh | sh")
        os.system("export OLLAMA_HOST=localhost:8888")
        os.system("sudo service ollama stop")
        os.system("ollama serve")
        os.system("ollama pull mxbai-embed-large")
        os.system("ollama pull jimscard/whiterabbit-neo")
        logging.info("Ollama setup completed successfully.")
    except Exception as e:
        logging.error(f"Error setting up Ollama: {e}")

def get_file_types(directory):
    file_types = set()

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            _, ext = os.path.splitext(filename)
            file_types.add(ext)
    
    return file_types

try:
    file_types = get_file_types(DATA_PATH)
    documents = []

    for file_type in file_types:
        print(f"Found file type: {file_type}")
except FileNotFoundError as e:
    print(e)

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
                {"context": retriever | streamlit_llama3.format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            response = rag_chain.invoke(query)
            sl.write(response)
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            sl.write("An error occurred while processing your query. Please check the logs.")


