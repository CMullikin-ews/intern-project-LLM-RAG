import streamlit as st
import streamlit_whiterabbit
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import BM25Retriever
from langchain.pipelines import RAGPipeline
from langchain import Document

def get_file_types(directory):
    streamlit_whiterabbit.get_file_types(directory)

def create_directory_loader(file_type, directory_path):
    streamlit_whiterabbit.create_directory_loader(file_type, directory_path)

def load_documents():
    streamlit_whiterabbit.load_documents()

def process_input(urls, question):
    model_local = Ollama(model="mxbai-embed-large")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = streamlit_whiterabbit.RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

def create_knowledgeBase():
    streamlit_whiterabbit.create_knowledgeBase()

def load_knowledgeBase():
    streamlit_whiterabbit.load_knowledgeBase()

def load_prompt():
    streamlit_whiterabbit.load_prompt()

def format_docs(docs):
    streamlit_whiterabbit.format_docs()

# Initialize Ollama LLM
def load_llm():
    """
    Creates and returns WhiteRabbitNeo model

    Returns:
        WhiteRabbitNeo: LLM
        """
    streamlit_whiterabbit.load_llm()
    return Ollama()

# Initialize document store
document_store = InMemoryDocstore()

# Add documents to the store (this is an example; replace with your documents)
documents = [
    Document({"content": "../cyber_data", "metadata": {"source": "Wikipedia"}}),
    # Add more documents as needed
]
document_store.write_documents(documents)

# Initialize retriever
retriever = BM25Retriever(document_store=document_store)

# Define a prompt template for RAG
prompt_template = PromptTemplate(
    template="Context: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

# Create an LLM chain for RAG
llm_chain = LLMChain(llm="Ollama", prompt_template=prompt_template)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(retriever=retriever, generator=llm_chain)

# Streamlit user interface
st.title("Advanced RAG System with LangChain and Ollama")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    # Retrieve and generate answer using RAG pipeline
    response = rag_pipeline({"question": question})
    st.write("Answer:", response)
