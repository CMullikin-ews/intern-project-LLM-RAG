# Packages used in RAG system
import streamlit as sl
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os

# Location of the documents for the vector store and location of the vector store
DATA_PATH = '../../cyber_data'
DB_FAISS_PATH = '../vectorstore'

# Specified loader for each type of file found in the cyber data directory (so far)
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

def get_file_types(directory):
        """
        Traverses all of the files in specified directory and returns types of files that it finds

        Args:
            directory (str): Path to directory

        Returns:
            Set[str]: All of the file types that can be found in the directory
        """
        file_types = set()

        for filename in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, filename)):
                        _, ext = os.path.splitext(filename)
                        file_types.add(ext)
        return file_types

def create_directory_loader(file_type, directory_path):
        """
        Creates and returns a DirectoryLoader using the loader specific to the file type provided
        
        Args:
            file_type (str): Type of file to make loader for
            directory_path (str): Path to directory

        Returns:
            DirectoryLoader: loader for the files in the directory provided
        """
        return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders.get(file_type, UnstructuredFileLoader)
)

def load_documents():
        """
        Loads in files from ../data directory and returns them

        Returns:
                List[Document]: Array of documents
        """
        file_types = get_file_types(DATA_PATH)
        documents = []
        
        for file_type in file_types:
                if file_type.strip() != "":
                        if file_type == '.json':
                                loader_list = create_directory_loader(file_type, DATA_PATH)
                                for loader in loader_list:
                                        docs = loader.load()
                                        chunks = split_text(docs)
                                        if chunks != None and chunks != "" and len(chunks) > 0:
                                                documents.extend(chunks)
                        else:        
                                loader = create_directory_loader(file_type, DATA_PATH)
                                docs = loader.load()
                                chunks = split_text(docs)
                                if chunks != None and chunks != "" and len(chunks) > 0:
                                        documents.extend(chunks)
        return documents

def split_text(docs, chunk_size=512, chunk_overlap=50):
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
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return chunks

def create_knowledgeBase():
        """
        Loads in documents, splits into chunks, and vectorizes chunks and stores vectors under FAISS vector store
        """
        documents = load_documents()
        embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings, allow_dangerous_deserialization=True)
        vectorstore.save_local(DB_FAISS_PATH)

def load_knowledgeBase():
        """
        Loads and returns vector store

        Returns:
            FAISS: vector store
        """
        embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
def load_llm():
        """
        Creates and returns WhiteRabbitNeo model

        Returns:
            WhiteRabbitNeo: LLM
        """
        llm = Ollama(model="jimscard/whiterabbit-neo")
        return llm

#creating prompt template using langchain
def load_prompt():
        """
        Creates and returns prompt for LLM query that specifies how response sounds and structure of response

        Returns:
            ChatPromptTemplate: Prompt for LLM
        """
        prompt = """
        You are an assistant for helping software developers to detect and neutralize viruses.
        Make sure to clearly define any necessary terms and go through the steps to use any application or software.
        Cite the documents that the data provided comes from and any other sources used.
        If the answer is not in the data provided answer "Sorry, I'm not sure how to respond to this"
        """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

def format_docs(docs):
        """
        Joins documents retrieved from vector store in one line format to make it easier for LLM to parse
        
        Args:
            docs (Document): Documents from vector stores

        Returns:
            String: documents in one line
        """
        return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
        # Downloads and runs ollama, as well as pulling our embedding model and LLM
        # os.system("curl -fsSL https://ollama.com/install.sh | sh")
        os.system("export OLLAMA_HOST=0.0.0.0")
        os.system("sudo service ollama stop")
        os.system("ollama serve")
        os.system("ollama pull mxbai-embed-large")
        
        # Creates header for streamlit app and writes to it
        sl.header("Welcome to the 📝Computer Virus copilot")
        sl.write("🤖 You can chat by entering your queries")
        
        # Creates and loads all of components for RAG system
        # create_knowledgeBase()
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        
        # Creates text box for user to query data
        query=sl.text_input('Enter some text')
        
        if(query):
                # Gets most similar vectors from knowledge base to user query and turns into actual documents
                similar_embeddings=knowledgeBase.similarity_search(query)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                
                # Defines chain together query, documents, prompt, and LLM to form process for generating response
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                # Calls chain and writes response to streamlit
                response=rag_chain.invoke(query)
                sl.write(response)
                