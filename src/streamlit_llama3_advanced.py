import os
import logging
import streamlit as st
import streamlit_llama3
import random
from langchain_community.document_loaders import WebBaseLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = Ollama(model='llama3')

# Initialize in-memory docstore with empty dictionary
docstore = InMemoryDocstore(docs={})

# Define the RAG pipeline
class RAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        # Use the retriever to get relevant documents
        documents = self.retriever.get_relevant_documents(query)
        # Process documents with the LLM
        response = self.llm.generate(query, documents)
        return response

pipeline = RAGPipeline(llm=llm, retriever=BM25Retriever(docs=docstore))

# Location of the documents for the vector store and location of the vector store
DATA_PATH = '../cyber_data'
DB_FAISS_PATH = '../vectorstore'

def create_document_loaders():
    """Create and return document loaders for different sources."""
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
    return loaders

def process_input(urls, question):
    model_local = Ollama(model="llama3")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = streamlit_llama3.RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

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

def get_relevant_url(query: str) -> List[str]:
    # Simulate getting a relevant URL
    urls = [
        "https://example.com/info1",
        "https://example.com/info2",
        "https://example.com/info3",
        "https://example.com/info4"
    ]
    url = random.choice(urls)
    return url

def respond_with_url(query: str) -> List[str]:
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
        st.warning("Vector store not found. Create it first.")

    # Assuming the following functions exist in streamlit_llama3
    try:
        knowledgeBase = streamlit_llama3.load_knowledgeBase()
        llm = streamlit_llama3.load_llm()
        prompt = streamlit_llama3.load_prompt()
    except AttributeError:
        st.error("Failed to load knowledge base or LLM components.")

    query = st.text_input('Enter some text')

    if query:
            similar_embeddings = knowledgeBase.similarity_search(query)
            documents = [Document(page_content=doc.page_content) for doc in similar_embeddings]
            retriever = FAISS.from_documents(documents=documents, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)).as_retriever()
            rag_chain = (
                {"context": retriever | streamlit_llama3.format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            response = rag_chain.invoke(query)
            st.write(response)
