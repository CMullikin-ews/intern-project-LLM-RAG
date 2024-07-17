#import Essential dependencies
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import random

# Specify the correct path to your documents directory
DATA_PATH = '../data'
DB_FAISS_PATH = '../vectorstore'

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

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

def load_knowledgeBase():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    llm = Ollama(model="llama3")
    return llm

def load_prompt():
    prompt = """
    You need to answer the question in the sentence as same as in the PDF content.
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    If the answer is not in the PDF, answer "Sorry, I'm not sure how to respond to this"
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
    response = generate_response(query)
    url = get_relevant_url(query)
    full_response = f"{response} For more information, visit: {url}"
    return full_response
# Example usage
user_query = "How does photosynthesis work?"
response_with_url = respond_with_url(user_query)
print(response_with_url)


if __name__ == '__main__':
    st.header("Welcome to the 📝 PDF Bot")
    st.write("🤖 You can chat by entering your queries")

    if not os.path.exists(DB_FAISS_PATH):
        create_knowledgeBase()

    knowledgeBase = load_knowledgeBase()
    llm = load_llm()
    prompt = load_prompt()

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
