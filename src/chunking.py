import nltk
import ollama
import numpy as np
import faiss
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import gradio as gr
from langchain.prompts import PromptTemplate

# Load the PDF
loader = PDFPlumberLoader("random machine learing pdf.pdf")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(OllamaEmbeddings())
documents = text_splitter.split_documents(docs)


# Instantiate the embedding model
embedder = OllamaEmbeddings()

# Create the vector store and fill it with embeddings
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define llm
llm = Ollama(model="mistral")

# Define the prompt
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None)
              
qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True)

def respond(question,history):
    return qa(question)["result"]


gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to Machine learning", container=False, scale=7),
    title="Machine learning Chatbot",
    examples=["What are different kinds of Machine learning capabilities", "What is AI"],
    cache_examples=True,
    retry_btn=None,

).launch(share = True)


def split_text_into_chunks(text, max_chunk_size=100):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

text = "Your long text goes here. It will be split into smaller chunks while maintaining sentence structure. This helps in ensuring the coherence of each chunk."

chunks = split_text_into_chunks(text, max_chunk_size=100)
for chunk in chunks:
    print(chunk)
    print("---")
