#import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import os
import shutil
from typing import List
from langchain.schema import Document
from transformers import GPT2Tokenizer

DATA_PATH = "random_machine_learning_pdf.pdf"
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    docs = loader.load()
    return docs

def split_text(text, max_length=512, chunk_overlap=50):
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
        max_length=max_length,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

# Example usage
text = "Your long text here."
split_texts = split_text(text, max_length=512, chunk_overlap=50)

for i, chunk in enumerate(split_texts):
    print(f"Chunk {i+1}:\n{chunk}\n")

def save_to_chroma(chunks: List[str]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    document_chunks = [Document(page_content=chunk) for chunk in chunks]
    db = Chroma.from_documents(document_chunks, embeddings, persist_directory=CHROMA_PATH)
    return db

#function to load the vectordatabase
def load_knowledgeBase():
    embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    DB_FAISS_PATH = '../vectorstore'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db
        
#function to load the OPENAI LLM
def load_llm():
    llm = Ollama(model="llama3")
    return llm

#creating prompt template using langchain
def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    if the answer is not in the pdf answer "Sorry, I'm not sure how to respond to this"
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__=='__main__':
        sl.header("welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by entering your queries ")
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        query=sl.text_input('Enter some text')
        
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                similar_embeddings=knowledgeBase.similarity_search(query)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                
                #creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                response=rag_chain.invoke(query)
                sl.write(response)
                
                