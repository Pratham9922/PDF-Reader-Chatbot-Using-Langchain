import streamlit as st
#use your openAIs API Key i had imported it from a file in my folder
from constant import api_key
import PyPDF2
import json

# import os
# import dill
# import pickle

import ssl, pickle
from PyPDF2 import PdfFileReader, PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

# from langchain.llms import OpenAI
# from langchain_openai import OpenAIEmbedding

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# load_dotenv()

# def read_pdf(pdf_file_path):
#     with open(pdf_file_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         num_pages = pdf_reader.numPages

#         text = ''
#         for page_num in range(num_pages):
#             page = pdf_reader.getPage(page_num)
#             text += page.extractText()

#     return text


def send_docs(file):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs

def send_chain(file,chain_type, k):
    docs=send_docs(file)
    # define embedding
    embeddings = OpenAIEmbeddings(openai_api_key= api_key)

    # create vector database from data
    # db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    db = Chroma.from_documents(docs, embeddings)
    system_template = """
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    ----------------
    {summaries}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # define retriever
    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(openai_api_key='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',model_name="gpt-3.5-turbo", temperature=0), 
    #     chain_type=chain_type, 
    #     retriever=retriever, 
    #     return_source_documents=True,
    #     return_generated_question=True,
    # )
    
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(openai_api_key='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',model_name="gpt-3.5-turbo", temperature=0), 
        chain_type=chain_type, 
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return qa 

def main():
    
    st.title("Chat with Data ❤️")
    # st.image("C:/Users/Pratham/Desktop/last_hope_chatbot/iitb_image.jpg")
    st.header("PlacementQueryBot")

    # Specify the absolute path to the PDF file
    pdf_file_path = 'C:/Users/Pratham/Downloads/student-placement-policy.pdf'

    # Usage
    # text_content = read_pdf(pdf_file_path)

    # user asking question
    query = st.text_input("Ask a question:")
    # st.write(query)


    #upload a your pdf file
    # pdf = st.file_uploader("Upload your PDF", type='pdf')
    # st.write(pdf.name)

    # if pdf is not None:
    #     pdf_reader = PdfReader(pdf)

    #     text = ""
    #     for page in pdf_reader.pages:
    #         text+= page.extract_text()

    #     #langchain_textspliter
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size = 1000,
    #         chunk_overlap = 200,
    #         length_function = len
    #     )

    #     chunks = text_splitter.split_text(text=text)

        
    #     #store pdf name
    #     store_name = pdf.name[:-4]
        
    #     if os.path.exists(f"{store_name}.pkl"):
    #         with open(f"{store_name}.pkl","rb") as f:
    #             vectorstore = dill.load(f)
    #         #st.write("Already, Embeddings loaded from the your folder (disks)")
    #     else:
    #         #embedding (Openai methods) 
    #         embeddings = OpenAIEmbeddings(openai_api_key='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')

    #         #Store the chunks part in db (vector)
    #         vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

    #         with open(f"{store_name}.pkl","wb") as f:
    #             dill.dump(vectorstore,f)
            
    #         #st.write("Embedding computation completed")

    #     #st.write(chunks)
        

    if query:

        # docs = vectorstore.similarity_search(query=query,k=3)
        #st.write(docs)
        
        #openai rank lnv process
        # llm = OpenAI(openai_api_key='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',temperature=0)
        # chain = load_qa_chain(llm=llm, chain_type= "stuff")

        docs=send_docs("C:/Users/Pratham/Downloads/student-placement-policy.pdf")
        qa=send_chain("C:/Users/Pratham/Downloads/student-placement-policy.pdf","stuff",k=3)
        
        # with get_openai_callback() as cb:
        #     response = qa.run(input_documents = docs, question = query)
        #     print(cb)
        # st.write(response)
        response = qa(query, return_only_outputs=True)
        # parsed_response=json.loads(response)
        # answer = parsed_response.get("answer", "")
        answer = response.get("answer", "")
        st.write(answer)



if __name__=="__main__":
    main()