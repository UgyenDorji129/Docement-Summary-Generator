from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def document_loader():
    path = input("\n\nENTER THE PATH OF THE FILES: ")
    try:
        if (path.endswith(".pdf")):
            loader = PyMuPDFLoader(path)
            pages = loader.load_and_split()
            # print(loader)
            # print("\n\nThe documents we have read is: \n\n",pages.page_content)
            # print("\nNumber of pages: ", len(pages))
            print("\nDocument Loaded!")
            return pages

        elif (path.endswith(".txt")):
            loader = TextLoader(path)
            pages = loader.load_and_split()
            print("\nDOCUMENT LOADED!")
            return pages

        elif (path.endswith(".docx")):
            loader = Docx2txtLoader(path)
            pages = loader.load_and_split()
            print("\nDOCUMENT LOADED!")
            return pages
        else:
            return None
    except:
        print("\nSOME ERROR OCCURED WHILE LOADING THE FILE, PLEASE TRY AGIAN")
        return None

def document_splitter(pages):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunck_text = text_splitter.split_documents(pages)
        # print("\n\n",chunck_text)
        print("\nDOCUMENT SPLITTED")
        return chunck_text
    except:
        print("\nSOME ERROR OCCURED WHILE SPLITTING THE DOCUMENT, PLEASE TRY AGAIN!")
        return None


def document_embedding(chunck_text):
    try:
        embeddings = OpenAIEmbeddings()
        vector_db = Milvus.from_documents(
            chunck_text,
            embeddings,
            connection_args={"host": os.getenv('HOST'), "port": os.getenv('PORT')},
            drop_old=True
        )
        print("\nDOCUMENT STORED IN MILVUS")
        return vector_db
    except:
        print('\nFACING SOME ISSUE WITH STORING THE DATA IN DB, PLEASE TRY AGAIN')
        return None


def start_conversation(vector_db):
    try:
        llm = OpenAI(temperature=0, top_p=0)
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_db.as_retriever(), memory=memory)

        print("\nWELCOME TO THE CHAT SECTION, ASK YOUR QUESTION OR PRESS ENTER ON THE EMPTY TO STOP SESSION\n")
        with get_openai_callback() as callback:
            while (True):
                query = input("\nEnter your question: ")
                if (query != ""):
                    result = qa({"question": query})
                    print("\nAnswer: ", result['answer'])
                else:
                    print("\nTHANK YOU FOR ASKING QUESTION!")
                    break
            print("\n\nTOKEN USAGE SUMMARY:")
            print(callback)
    except:
        print("\nFACING SOME ISSUE FOR NOW, PLEASE TRY AGAIN!")


if __name__ == "__main__":
    try:
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        pages = document_loader()
        chunck_text = document_splitter(pages=pages)
        vector_db = document_embedding(chunck_text=chunck_text)
        start_conversation(vector_db=vector_db)
    except:
        print("\nSOME ERROR OCCURED, PLEASE TRY AGAIN!")
