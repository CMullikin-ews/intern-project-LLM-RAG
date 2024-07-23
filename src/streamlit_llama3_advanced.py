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
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Downloads and runs Ollama, as well as pulling our embedding model and LLM
def setup_ollama():
    try:
        os.system("curl -fsSL https://ollama.com/install.sh | sh")
        os.system("export OLLAMA_HOST=localhost:8888")
        os.system("sudo pkill ollama")
        os.system("ollama serve")
        os.system("ollama pull mxbai-embed-large")
        os.system("ollama pull jimscard/whiterabbit-neo")
        logging.info("Ollama setup completed successfully.")
    except Exception as e:
        logging.error(f"Error setting up Ollama: {e}")

# Define data paths
DATA_PATH = '../cyber_data'
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

# Function to get file types
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
            if file_type.strip() != "":
                    if file_type == '.json':
                            loader_list = create_directory_loader(file_type, DATA_PATH)
                            for loader in loader_list:
                                    docs = loader.load()
                                    chunks = split_text(docs)
                                    if chunks != None and chunks != "" and len(chunks) > 0:
                                            documents.extend(chunks)
                    else:        
                            loader = create_directory_loader(file_type, DATA_PATH)
                            docs = loader.load()
                            chunks = split_text(docs)
                            if chunks != None and chunks != "" and len(chunks) > 0:
                                    documents.extend(chunks)
    return documents


def split_text(docs, max_length=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc.page_content))
    return chunks


def create_knowledgeBase():
    docs = load_documents()
    chunks = split_text(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

# Load knowledge base
def load_knowledgeBase():
    file_types = get_file_types(DATA_PATH)
    document_loaders = [loaders[file_type](os.path.join(DATA_PATH, f'*.{file_type}')) for file_type in file_types if file_type in loaders]
    documents = []
    for loader in document_loaders:
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, OllamaEmbeddings(model="mxbai-embed-large"))
    vector_store.save(DB_FAISS_PATH)
    return vector_store

# Load LLM
def load_llm():
    return Ollama(model="jimscard/whiterabbit-neo", show_progress=True)

# Load prompt
def load_prompt():
    prompt_text = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    {context}
    Question: {question}
    """
    return ChatPromptTemplate.from_template(prompt_text)

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    setup_ollama()
    
    # Creates header for Streamlit app and writes to it
    sl.header("Welcome to the 📝PDF bot")
    sl.write("🤖 You can chat by entering your queries")
    
    # Load components for RAG system
    try:
        knowledge_base = load_knowledgeBase()
        llm = load_llm()
        prompt = load_prompt()
        logging.info("Components loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading components: {e}")
        sl.write("An error occurred while loading the components. Please check the logs.")

    # Create text box for user to query data
    query = sl.text_input('Enter some text')
    
    if query:
        try:
            similar_embeddings = streamlit_llama3.knowledge_base.similarity_search(query)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
            
            # Define the chain for generating response
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Generate response and write to Streamlit
            response = rag_chain.invoke(query)
            sl.write(response)
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            sl.write("An error occurred while processing your query. Please check the logs.")
