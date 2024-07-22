import streamlit as st
from langchain import LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ollama import Ollama
from langchain.document_store import InMemoryDocumentStore
from langchain.retrievers import BM25Retriever
from langchain.pipeline import RAGPipeline

# Initialize LangChain
lc = LangChain()

# Initialize Ollama LLM
ollama_llm = Ollama(api_key='YOUR_API_KEY')

# Register the LLM with LangChain
lc.register_llm(ollama_llm)

# Initialize document store
document_store = InMemoryDocumentStore()

# Add documents to the store (this is an example; replace with your documents)
documents = [
    {"content": "../cyber_data", "metadata": {"source": "Wikipedia"}},
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
llm_chain = LLMChain(llm=ollama_llm, prompt_template=prompt_template)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(retriever=retriever, generator=llm_chain)

# Streamlit user interface
st.title("Advanced RAG System with LangChain and Ollama")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    # Retrieve and generate answer using RAG pipeline
    response = rag_pipeline.run({"question": question})
    st.write("Answer:", response)
