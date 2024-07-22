import os
import logging
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import Ollama
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import streamlit_llama3  # Fixed import position

DATA_PATH = '../../cyber_data'
DB_FAISS_PATH = '../vectorstore'

# Initialize Ollama LLM
llm = Ollama(model='llama3')

# Initialize document store
def load_documents():
    streamlit_llama3.load_documents()

def split_text(docs, max_length=512, chunk_overlap=50):
    streamlit_llama3.split_text(docs, max_length=512, chunk_overlap=50)

def create_knowledgeBase():
    streamlit_llama3.create_knowledgeBase()

def load_prompt():
    streamlit_llama3.load_prompt()

def format_docs(docs):
    streamlit_llama3.format_docs(docs)
    
document_store = FAISS(
    faiss_index_path=DATA_PATH, 
    faiss_config_path=DB_FAISS_PATH
)

# Add documents to the store (this is an example; replace with your documents)
documents = [
    {"content": DATA_PATH, "meta": {"source": "Wikipedia"}},  # Update content as needed
    # Add more documents as needed
]
document_store.write_documents(documents)

# Initialize retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True
)

# Update embeddings for the documents in the store
document_store.update_embeddings(retriever)

class RAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        documents = self.retriever.retrieve(query)
        # Process documents with the LLM
        response = self.llm.generate(query, documents)
        return response

pipeline = RAGPipeline(llm=llm, retriever=retriever)

# Define a prompt template for RAG
prompt_template = PromptTemplate(
    template="Context: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

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
            {"context": retriever | streamlit_llama3.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        st.write(response)