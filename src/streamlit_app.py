# %% [markdown]
# mport Essential dependencies

# %%
import streamlit as st
import nltk
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import GPT2Tokenizer
# %% [markdown]
# unction to load the vectordatabase

# %%
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=os.environ['API_key'] )
        DB_FAISS_PATH = '../vectorstore'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.environ['API_key'] )
        return llm

# %% [markdown]
# reating prompt template using langchain

# %%
def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf answer "Sorry, I'm not sure how to respond to this"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %%
def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    return len(text)  # Simplified for this example

def split_text_into_chunks(text, token_limit):
    if count_tokens(text) <= token_limit:
        return [text]
    
    # Find the midpoint to split the text
    midpoint = len(text) // 2
    
    # Split the text at the midpoint
    left_part = text[:midpoint]
    right_part = text[midpoint:]
    
    # Ensure we split at a word boundary to avoid cutting words in half
    while not left_part.endswith(' ') and midpoint > 0:
        midpoint -= 1
        left_part = text[:midpoint]
        right_part = text[midpoint:]
    
    # Recursively split the text parts
    left_chunks = split_text_into_chunks(left_part.strip(), token_limit)
    right_chunks = split_text_into_chunks(right_part.strip(), token_limit)
    
    return left_chunks + right_chunks

# Example usage
text = "This is a very long text that needs to be split into smaller chunks. Each chunk should not exceed the token limit."
token_limit = 10

chunks = split_text_into_chunks(text, token_limit)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")

index = {
    'this': [1],
    'is': [1],
    'the': [1],
    'first': [1],
    'document': [1],
    'second': [2],
    'for': [2],
    'testing': [2],
    'purposes': [2],
    'another': [3],
    'example': [3]
}

def query_llm(query, index):
    tokens = word_tokenize(query.lower())  # Tokenize and convert to lowercase
    results = set(index.get(token, []) for token in tokens)  # Get document IDs for each token
    # Flatten the list of lists and return unique document IDs
    return sorted(set(doc_id for sublist in results for doc_id in sublist))

# Example usage:
query = "example document"
query_result = query_llm(query, index)
print("Query:", query)
print("Matching Documents:", query_result)

def index_documents(documents):
    index = defaultdict(list)
    for doc_id, doc_text in documents.items():
        tokens = word_tokenize(doc_text.lower())  # Tokenize and convert to lowercase
        for token in tokens:
            index[token].append(doc_id)
    
    return index
# Example documents
documents = {
    1: "This is the first document.",
    2: "Second document for testing purposes.",
    3: "Another example document."
}

# Indexing the documents
index = index_documents(documents)

# Example: Retrieve documents containing the token "document"
print(index["document"])  # Output: [1, 2, 3]

if __name__=='__main__':
        st.header("welcome to the 📝PDF bot")
        text = st.empty()
        box = st.empty()
        text.write("Please enter your API key: ")
        API_key = box.text_input('Enter API key')

        if API_key:
                os.environ['API_key'] = API_key
                text.write("🤖 You can chat by Entering your queries ")
                query=box.text_input('Enter some text')
        
                if(query):
                        knowledgeBase=load_knowledgeBase()
                        llm=load_llm()
                        prompt=load_prompt()

                        #getting only the chunks that are similar to the query for llm to produce the output
                        similar_embeddings=knowledgeBase.similarity_search(query)
                        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=os.environ['API_key'] ))
                        
                        #creating the chain for integrating llm,prompt,stroutputparser
                        retriever = similar_embeddings.as_retriever()
                        rag_chain = (
                                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                                | prompt
                                | llm
                                | StrOutputParser()
                        )
                        response=rag_chain.invoke(query)
                        user_response = st.empty()
                        user_response.write(response)
        


