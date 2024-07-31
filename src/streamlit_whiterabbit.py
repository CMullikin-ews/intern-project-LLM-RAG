# Packages used in RAG system
import streamlit as sl
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import DoctranPropertyExtractor
import logging
import os
import pathlib
import subprocess

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

logging.basicConfig(level=logging.INFO, filename = 'vector_log.log', filemode = 'w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_ollama():
        """
        Downloads (if necessary) and runs ollama locally
        """
        # os.system("curl -fsSL https://ollama.com/install.sh | sh")
        # os.system("export OLLAMA_HOST=localhost:8888")
        os.system("sudo service ollama stop")
        cmd = "ollama serve"
        with open(os.devnull, 'wb') as devnull:
            process = subprocess.Popen(cmd, shell=True, stdout=devnull, stderr=devnull)

def txt_file_rename(directory):
    """
    Takes .txt files and renames them if they have a line containing title in them

    Args:
        directory (str): path to directory where files are stored
    """
    file_paths = pathlib.Path(directory).glob('*.txt')
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                segments = line.split(':')
                if 'title' in segments[0].lower() and len(segments) >= 2:
                    name = segments[1].strip()
                    new_file_name = os.path.join(directory, name + file_ext)
                    try:
                        os.rename(file_path, new_file_name)
                        print(f'Renamed {file_name} to {name}')
                    except FileNotFoundError:
                        print(f"FileNotFoundError: {file_path} not found.")
                    except PermissionError:
                        print("Permission denied: You don't have the necessary permissions to change the permissions of this file.")
                    except NotADirectoryError:
                        print(f"Not a directory: {new_file_name}")

def rename_files_in_directory(directory):
    """
    Renames all the .md files in the directory with several periods in them to dashes

    Args:
        directory (str): path to directory
    """
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            parts = filename.rsplit('.', 2)
            if len(parts) == 3 and parts[1].isdigit():
                new_filename = f"{parts[0]}-{parts[1]}.md"
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                print(f"Renamed: {filename} -> {new_filename}")

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
        if file_type == '.json':
            loader_list = []
            for file_name in [file for file in os.listdir(directory_path) if file.endswith('.json')]:
                loader_list.append(JSONLoader(file_path=directory_path+'/'+file_name, jq_schema='.', text_content=False))
            return loader_list
        else:
            return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders.get(file_type, UnstructuredFileLoader))

def split_text(docs, chunk_size=512, chunk_overlap=64):
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
        logger.info(f"Chunks created: {len(chunks)}")
        return chunks
    
# def metadata_extractor(documents):
#     properties = [
#     {
#         "name": "category",
#         "description": "What type of document this is.",
#         "type": "string",
#         "enum": ["code_block", "instructions", "explanation"],
#         "required": True,
#     },
#     {
#         "name": "malware",
#         "description": "A list of all malware mentioned in this document.",
#         "type": "array",
#         "items": {
#             "name": "computer_malware",
#             "description": "The full name of the malware used",
#             "type": "string",
#         },
#         "required": True,
#     },
#     {
#         "name": "eli5",
#         "description": "Explain this email to me like I'm 5 years old.",
#         "type": "string",
#         "required": True,
#     },
# ]
    
#     property_extractor = DoctranPropertyExtractor(properties=properties)
#     extracted_document = property_extractor.transform_documents(documents, properties=properties)
#     return extracted_document

def chunk_numberer(docs):
    """
    Numbers each chunk in each document to make it easier to find where chunks are located in documents

    Args:
        docs (List[Document]): list of documents 

    Returns:
        List[Document]: list of documents 
    """
    num = 1
    if docs:
        source = docs[0].metadata.get('source', 'unknown')
        for doc in docs:
            if 'source' not in doc.metadata:
                logger.error(f"Missing 'source' in document metadata: {doc.metadata}")
                continue
            if source != doc.metadata['source']:
                num = 1
                source = doc.metadata['source']
            doc.metadata['chunk_no'] = num
            num += 1
    return docs

def load_documents(directory):
        """
        Loads in files from ../data directory and returns them

        Returns:
                List[Document]: Array of documents
        """
        file_types = get_file_types(directory)
        documents = []
        
        for file_type in file_types:
                if file_type.strip() != "":
                        if file_type == '.json':
                                loader_list = create_directory_loader(file_type, directory)
                                for loader in loader_list:
                                        docs = loader.load()
                                        chunks = split_text(docs)
                                        if chunks != None and chunks != "" and len(chunks) > 0:
                                                documents.extend(chunks)
                        else:        
                                loader = create_directory_loader(file_type, directory)
                                docs = loader.load()
                                chunks = split_text(docs)
                                if chunks != None and chunks != "" and len(chunks) > 0:
                                        documents.extend(chunks)
        documents = chunk_numberer(documents)
        return documents
        # return metadata_extractor(documents)

def create_knowledgeBase(directory, vectorstore):
    """
    Loads in documents, splits into chunks, and vectorizes chunks and stores vectors under FAISS vector store
    
    Parameters:
        directory (str): The input text to be split.
        vectorstore (FAISS): vector store containing vectors of documents
    """
    documents = load_documents(directory)
    os.system("ollama pull mxbai-embed-large")
    embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
    if len(documents) > 0:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        if os.path.exists(DB_FAISS_PATH + '/index.faiss'):
            old_vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            old_vectorstore.merge_from(vectorstore)
            old_vectorstore.save_local(DB_FAISS_PATH)
        else:
            vectorstore.save_local(DB_FAISS_PATH)

def move_files(directory, new_directory):
    """
    Moves files from unprocessed data directory to processed data directory
    
    Parameters:
        directory (str): The input text to be split.
    """
    file_paths = pathlib.Path(directory).iterdir()
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        os.replace(file_path, new_directory+file_name)

def load_knowledgeBase():
        """
        Loads and returns vector store

        Returns:
            FAISS: vector store
        """
        os.system("ollama pull mxbai-embed-large")
        embeddings=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
def load_llm():
        """
        Creates and returns WhiteRabbitNeo model

        Returns:
            WhiteRabbitNeo: LLM
        """
        os.system("ollama pull jimscard/whiterabbit-neo")
        llm = Ollama(model="jimscard/whiterabbit-neo")
        return llm

def load_prompt():
        """
        Creates and returns conda install -c pytorch faiss-gprompt for LLM query that specifies how response sounds and structure of response

        Returns:
            ChatPromptTemplate: Prompt for LLM
        # """
        # prompt = """You need to answer the question in the sentence as same as in the content.
        # Cite the sources of any data provided.
        # Given below is the context and question of the user.
        # context = {context}
        # question = {question}
        # if the answer is not in the data provided answer "Sorry, I'm not sure how to respond to this."
        # """
        
        prompt = """Provide detailed steps to address the question about offensive cyber operations.
        Cite the sources of any data provided.
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the data provided answer "Sorry, I'm not sure how to respond to this."
        """
    
        # prompt = """Given the following question and context, extract any part of the 
        # context that is relevant to answer the question. 
        # Cite the sources of any data provided.
        # Given below is the context and question of the user.
        # context = {context}
        # question = {question}
        # if the answer is not in the data provided answer "Sorry, I'm not sure how to respond to this."
        # """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

def load_reranker():
        """
        Creates and returns MixedBread reranker algorithm

        Returns:
            MixedBread: reranker
        """
        reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")
        return reranker    

def format_docs(docs):
        """
        Logs and joins documents retrieved from vector store in one line format to make it easier for LLM to parse
        
        Args:
            docs (Document): Documents from vector stores

        Returns:
            String: documents in one line
        """        
        reranker = load_reranker()
        
        docs_content = []
        for doc in docs:
                logger.info(f"\nDocument used in query for {query}: {doc}")
                docs_content.append(str(doc.page_content))
                
        ranked_docs = reranker.rank(query, docs_content, return_documents=True)
        ranked_docs_content = []
        for ranked_doc in ranked_docs:
                ranked_docs_content.append(str(ranked_doc.get('text')))
        
        return "\n\n".join(ranked_docs_content)

def load_compressor():
        """
        Creates and returns contextual compressor using LLM which reduces size of documents from vector store

        Returns:
            LLMChainExtractor: contextual compressor
        """
        llm = load_llm()
        compressor = LLMChainExtractor.from_llm(llm)
        return compressor


def respond_with_sources(query, retriever) -> str:
    """
    Pulls and returns the sources of all the documents used in the query

    Args:
        query (str): query inputted by user
        retriever (FAISS_retriver): gets most similar vectors from vector store

    Returns:
        str: each source used
    """
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        logger.info("No documents retrieved for query.")
        return "No documents retrieved."
    sources = {doc.metadata['source'].replace('/', '.').split('.')[-2] + f" (chunk {doc.metadata['chunk_no']})" for doc in retrieved_docs}
    citation_text = "Documents used: " + ", ".join(sources)
    return f"\n\n{citation_text}"

if __name__=='__main__':
        #Pulls and serves ollama for models used later on
        setup_ollama()
        
        # Location of the documents for the vector store and location of the vector store
        DATA_PATH = '../../unprocessed_cyber_data'
        NEW_DATA_PATH = '../../processed_cyber_data/'
        DB_FAISS_PATH = '../vectorstore'

        
        # Creates header for streamlit app and writes to it
        sl.header("Welcome to the 📝 Offensive Cyber Assistant")
        sl.write("🤖 You can chat by entering your queries")
        
        try:
                #Creates vector store using any unprocessed files
                rename_files_in_directory(DATA_PATH)
                txt_file_rename(DATA_PATH)
                # create_knowledgeBase(DATA_PATH, DB_FAISS_PATH)
                move_files(DATA_PATH, NEW_DATA_PATH)
                
                # Loads in vector store, LLM, and prompt
                knowledge_base = load_knowledgeBase()
                llm = load_llm()
                prompt = load_prompt()
                logger.info("Components loaded successfully.")
                
                # Creates text box for user to query data
                query=sl.text_input('Enter some text')
                
                if(query):
                        # Gets most similar vectors from knowledge base to user query and turns into actual documents
                        similar_embeddings=knowledge_base.similarity_search(query)
                        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True))
                        
                        # Defines retriever for getting vectors from vector store
                        retriever = similar_embeddings.as_retriever()
                        compressor = load_compressor()
                        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
                        
                        # Chain that combines query, vectors, prompt, and LLM to generate response
                        rag_chain = (
                                {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                        
                        # Calls chain and writes response to streamlit
                        response=rag_chain.invoke(query) + respond_with_sources(query, retriever)
                        sl.write(response)
        
        except Exception as e:
            logger.error(f"\nError loading components: {e}")
            sl.write("An error occurred while loading the components. Please check the logs.")