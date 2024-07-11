#import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=os.environ['API_key'] )
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.environ['API_key'])
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
        text = sl.empty()
        box = sl.empty()
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
                        user_response = sl.empty()
                        user_response.write(response)
        
        
        
        