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
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


DATA_PATH = '../../cyber_data'
DB_FAISS_PATH = '../vectorstore'

# Initialize Ollama LLM
llm = Ollama(model='llama3')

# Initialize document store
document_store = FAISSDocumentStore(faiss_index_path="my_index.faiss")

# Add documents to the store (this is an example; replace with your documents)
documents = [
    {"content": "../cyber_data", "metadata": {"source": "Wikipedia"}},
    # Add more documents as needed
]
document_store.write_documents(documents)

# Initialize retriever
retriever = BM25Retriever(document_store=document_store)
class RAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        documents = self.retriever.get_relevant_documents(query)
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
