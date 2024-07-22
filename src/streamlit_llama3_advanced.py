import os
import logging
import streamlit as st
import streamlit_llama3
import random
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.retrievers import CustomRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = Ollama(model='llama3')
retriever = CustomRetriever()  # Initialize your custom retriever
pipeline = RAGPipeline(llm=llm, retriever=retriever)

# Location of the documents for the vector store and location of the vector store
DATA_PATH = '../cyber_data'
DB_FAISS_PATH = '../vectorstore'

def get_file_types(directory):
    streamlit_llama3.get_file_types(directory)

def create_directory_loader(file_type, directory_path):
    streamlit_llama3.create_directory_loader(file_type, directory_path)

def create_document_loaders():
    """Create and return document loaders for different sources."""
    loaders = {
    '.php': UnstructuredFileLoader,
    '.cs': UnstructuredFileLoader,
    '': UnstructuredFileLoader,
    '.c': UnstructuredFileLoader,
    '.html': UnstructuredHTMLLoader,
    '.md': UnstructuredMarkdownLoader,
    '.tzt': UnstructuredFileLoader,
    '.java': UnstructuredFileLoader,
    '.txt': TextLoader,
    '.ps1': UnstructuredFileLoader,
    '.delphi': UnstructuredFileLoader,
    '.asm': UnstructuredFileLoader,
    '.TXT': TextLoader
    }
    return loaders

def load_documents():
    streamlit_llama3.load_documents()

def process_input(urls, question):
    model_local = Ollama(model="llama3")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = streamlit_llama3.RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

def create_knowledgeBase():
    streamlit_llama3.create_knowledgeBase()

def load_knowledgeBase():
    streamlit_llama3.load_knowledgeBase()

def load_prompt():
    streamlit_llama3.load_prompt()

def format_docs(docs):
    streamlit_llama3.format_docs(docs)

def load_llm():
    streamlit_llama3.load_llm()

def generate_response(query):
    # Simulate an LLM response
    responses = [
        "Sure, I can help with that!",
        "Let me find that information for you.",
        "Here is what I found.",
        "This is the information you requested."
    ]
    response = random.choice(responses)
    return response

def get_relevant_url(query):
    # Simulate getting a relevant URL
    urls = [
        "https://example.com/info1",
        "https://example.com/info2",
        "https://example.com/info3",
        "https://example.com/info4"
    ]
    url = random.choice(urls)
    return url

def respond_with_url(query):
    retrieved_docs = retriever.retrieve(query)
    sources = [doc.metadata['source'] for doc in retrieved_docs]
    
    # Generate a response
    response = pipeline.generate(query)
    
    # Append citations to the response
    citation_text = "Sources: " + ", ".join(sources)
    response_with_citations = f"{response}\n\n{citation_text}"
    
    return response_with_citations


if __name__ == '__main__':
    st.header("Welcome to the 📝 PDF Bot")
    st.write("🤖 You can chat by entering your queries")

    if not os.path.exists(DB_FAISS_PATH):
        streamlit_llama3.create_knowledgeBase()

    knowledgeBase = streamlit_llama3.load_knowledgeBase()
    llm = streamlit_llama3.load_llm()
    prompt = streamlit_llama3.load_prompt()

    query = st.text_input('Enter some text')

    if query:
        similar_embeddings = knowledgeBase.similarity_search(query)
        documents = [Document(page_content=doc.page_content) for doc in similar_embeddings]
        retriever = FAISS.from_documents(documents=documents, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)).as_retriever()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        st.write(response)
