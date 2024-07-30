import pathlib
import subprocess
import streamlit as st
from mixedbread_ai_haystack.rerankers import MixedbreadAIReranker
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from sentence_transformers.cross_encoder import CrossEncoder
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_community.document_transformers import DoctranPropertyExtractor
import os
import logging
import random

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
    '.TXT': TextLoader,
    '.json': JSONLoader
}

# Setup logging
logging.basicConfig(level=logging.INFO, filename='vector_log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_ollama():
    """Downloads (if necessary) and runs ollama locally."""
    # os.system("curl -fsSL https://ollama.com/install.sh | sh")
    # os.system("export OLLAMA_HOST=localhost:8501")
    os.system("sudo service ollama stop")
    cmd = "ollama serve"
    with open(os.devnull, 'wb') as devnull:
        process = subprocess.Popen(cmd, shell=True, stdout=devnull, stderr=devnull)

def txt_file_rename(directory):
    """
    Takes .txt files and renames them if they have a line containing title in them

    Args:
        directory (str): path to directory where files are stored
    """
    file_paths = pathlib.Path(directory).glob('*.txt')
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                segments = line.split(':')
                if 'title' in segments[0].lower() and len(segments) >= 2:
                    name = segments[1].strip()
                    new_file_name = os.path.join(directory, name + file_ext)
                    try:
                        print(f'Renamed {file_name} to {name}')
                        os.rename(file_path, new_file_name)
                    except FileNotFoundError:
                        print(f"FileNotFoundError: {file_path} not found.")
                    except PermissionError:
                        print("Permission denied: You don't have the necessary permissions to change the permissions of this file.")
                    except NotADirectoryError:
                        print(f"Not a directory: {new_file_name}")

def load_reranker():
    """
    Creates and returns MixedBread reranker algorithm

    Returns:
        MixedBread: reranker
    """
    reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")
    return reranker   

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
    if file_type == '.json':
        loader_list = []
        for file_name in [file for file in os.listdir(directory_path) if file.endswith('.json')]:
            loader_list.append(JSONLoader(file_path=os.path.join(directory_path, file_name), jq_schema='.', text_content=False))
        return loader_list
    else:
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders.get(file_type, UnstructuredFileLoader))

def split_text(docs, chunk_size=512, chunk_overlap=64):
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
    
def metadata_extractor(documents):
    properties = [
    {
        "name": "category",
        "description": "What type of document this is.",
        "type": "string",
        "enum": ["code_block", "instructions", "explanation"],
        "required": True,
    },
    {
        "name": "malware",
        "description": "A list of all malware mentioned in this document.",
        "type": "array",
        "items": {
            "name": "computer_malware",
            "description": "The full name of the malware used",
            "type": "string",
        },
        "required": True,
    },
    {
        "name": "eli5",
        "description": "Explain this email to me like I'm 5 years old.",
        "type": "string",
        "required": True,
    },
]
    
    property_extractor = DoctranPropertyExtractor(properties=properties)
    extracted_document = property_extractor.transform_documents(documents, properties=properties)
    return extracted_document

def load_documents(directory):
        """
        Loads in files from ../data directory and returns them

        Returns:
                List[Document]: Array of documents
        """
        file_types = get_file_types(directory)
        documents = []
        
        for file_type in file_types:
                if file_type.strip() != "":
                        if file_type == '.json':
                                loader_list = create_directory_loader(file_type, directory)
                                for loader in loader_list:
                                        docs = loader.load()
                                        chunks = split_text(docs)
                                        if chunks != None and chunks != "" and len(chunks) > 0:
                                                documents.extend(chunks)
                        else:        
                                loader = create_directory_loader(file_type, directory)
                                docs = loader.load()
                                chunks = split_text(docs)
                                if chunks != None and chunks != "" and len(chunks) > 0:
                                        documents.extend(chunks)
        return documents


def create_knowledgeBase(directory, vectorstore):
    """
    Loads in documents, splits into chunks, and vectorizes chunks and stores vectors under FAISS vector store
    """
    documents = load_documents(directory)
    os.system("ollama pull mxbai-embed-large")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    if os.path.exists(DB_FAISS_PATH + '/index.faiss'):
        old_vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
        old_vectorstore.merge_from(vectorstore)
        old_vectorstore.save_local(DB_FAISS_PATH)
    else:
        vectorstore.save_local(DB_FAISS_PATH)

def move_files(directory):
    file_paths = pathlib.Path(directory).glob('*.txt')
    new_path = '../../processed_cyber_data'
    for file_path in file_paths:
        new_path = '../../processed_cyber_data/'
        file_name = os.path.basename(file_path)
        new_path += file_name
        os.replace(file_path, new_path)

def load_knowledgeBase():
    os.system("ollama pull mxbai-embed-large")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    os.system("ollama pull llama3")
    llm = Ollama(model="llama3")
    return llm

def load_prompt():
    prompt = """
    You are an assistant for helping software developers to detect and neutralize viruses.
    Make sure to clearly define any necessary terms and go through the steps to use any application or software.
    Only use the data provided to you.
    Cite the sources used in constructing the response.
    If the answer is not in the data provided, answer "Sorry, I'm not sure how to respond to this"
    """
    return ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    reranker = load_reranker()
        
    docs_content = []
    for doc in docs:
        logger.info(f"\nDocument used in query for {query}: {doc}")
        docs_content.append(str(doc.page_content))
                
    ranked_docs = reranker.rank(query, docs_content, return_documents=True)
    ranked_docs_content = []
    for ranked_doc in ranked_docs:
        ranked_docs_content.append(str(ranked_doc.get('text')))
        
    return "\n\n".join(ranked_docs_content)


def load_compressor():
    """
    Creates and returns contextual compressor using LLM which reduces size of documents from vector store

    Returns:
        LLMChainExtractor: contextual compressor
    """
    llm = load_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    return compressor

def generate_response(query: str) -> str:
    responses = [
        "Sure, I can help with that!",
        "Let me find that information for you.",
        "Here is what I found.",
        "This is the information you requested."
    ]
    return random.choice(responses)

def get_relevant_url(query: str) -> str:
    urls = [
        "https://example.com/info1",
        "https://example.com/info2",
        "https://example.com/info3",
        "https://example.com/info4"
    ]
    return random.choice(urls)

def respond_with_sources(query, retriever) -> str:
    # This function should be updated as per your logic to retrieve documents
    # As it stands, it assumes `retriever` is a global variable
    retrieved_docs = retriever.invoke(query)
    sources = {doc.metadata['source'].replace('/', '.').split('.')[-2] for doc in retrieved_docs}
    citation_text = "Documents used: " + ", ".join(sources)
    return f"\n\n{citation_text}"

if __name__ == '__main__':
    setup_ollama()

    DATA_PATH = '../../processed_cyber_data'
    DB_FAISS_PATH = '../vectorstore'

    st.header("Welcome to the 📝 Offensive Cyber Assistant")
    st.write("🤖 You can chat by entering your queries")

    try:
        # Creates vector store using any unprocessed files
        txt_file_rename(DATA_PATH)
        create_knowledgeBase(DATA_PATH, DB_FAISS_PATH)
        move_files(DATA_PATH)
        
        knowledge_base = load_knowledgeBase()
        llm = load_llm()
        prompt = load_prompt()
        logger.info("Components loaded successfully.")
        
        query = st.text_input('Enter some text')
    
        if query:
                similar_embeddings = knowledge_base.similarity_search(query)
                documents = [Document(page_content=doc.page_content) for doc in similar_embeddings]
                similar_embeddings = FAISS.from_documents(documents=documents, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                retriever = similar_embeddings.as_retriever()
                compressor = load_compressor()
                compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
                rag_chain = (
                    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                response = rag_chain.invoke(query) + respond_with_sources(query, retriever)
                st.write(response)

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        st.write("An error occurred while processing your query. Please check the logs.")
