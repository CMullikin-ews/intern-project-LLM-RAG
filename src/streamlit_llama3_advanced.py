import os
import logging
import streamlit as st
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
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = Ollama(model='llama3')

class RAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        documents = self.retriever.get_relevant_documents(query)
        # Process documents with the LLM
        response = self.llm.generate(query, documents)
        return response
    
retriever = BM25Retriever()
pipeline = RAGPipeline(llm=llm, retriever=retriever)

# Location of the documents for the vector store and location of the vector store
DATA_PATH = '../cyber_data'
DB_FAISS_PATH = '../vectorstore'

def create_document_loaders():
    """Create and return document loaders for different sources."""
    loaders = {
        '.php': UnstructuredFileLoader,
        '.cs': UnstructuredFileLoader,
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

def load_documents(directory_path):
    loaders = create_document_loaders()
    documents = []
    for ext, loader_cls in loaders.items():
        loader = loader_cls(directory_path)
        documents.extend(loader.load())
    return documents

def process_input(urls, question):
    model_local = Ollama(model="llama3")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    # Assuming further processing is needed for the doc_splits

def generate_response(query: str) -> List[str]:
    # Simulate an LLM response
    responses = [
        "Sure, I can help with that!",
        "Let me find that information for you.",
        "Here is what I found.",
        "This is the information you requested."
    ]
    response = random.choice(responses)
    return response

def respond_with_url(query: str) -> str:
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
        # Create knowledge base if it doesn't exist
        documents = load_documents(DATA_PATH)
        embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save(DB_FAISS_PATH)
    else:
        vectorstore = FAISS.load(DB_FAISS_PATH)

    query = st.text_input('Enter some text')

    if query:
        similar_embeddings = vectorstore.similarity_search(query)
        documents = [Document(page_content=doc.page_content) for doc in similar_embeddings]
        retriever = FAISS.from_documents(documents=documents, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)).as_retriever()
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PromptTemplate(prompt="Your query: {question}\n\n{context}")
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        st.write(response)


