from flask import Flask, request
import openai
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate


def document_loader(path):
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
        print("\nDocument Loaded!")
        return pages

    elif (path.endswith(".docx")):
        loader = Docx2txtLoader(path)
        pages = loader.load_and_split()
        print("\nDocument Loaded!")
        return pages
    else:
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
        print("\nDocument Splited!")
        return chunck_text
    except:
        return None


def document_embedding(chunck_text):
    try:
        embeddings = OpenAIEmbeddings()
        vector_db = Milvus.from_documents(
            chunck_text,
            embeddings,
            connection_args={"host": "localhost", "port": "19530"},
            drop_old=True
        )
        print("\nDocument Embedded and Stored in Malvus!")
        return vector_db
    except:
        return None


def summarize(pages):
    try:
        llm = OpenAI(temperature=0, top_p=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(input_documents=pages)
        print("\nSummary Generated!")
        return summary
    except:
        return None


app = Flask(__name__)


@app.route('/summarize_doc', methods=['POST'])
def summarize_doc():
    try:
        data = request.get_json()
        pages = document_loader(path=data['path'])
        summary = summarize(pages=pages)
        summary = {"result": summary}
        return summary
    except:
        print("Some error occured!")
        return None


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    app.run()
    # load_dotenv()
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    # pages = document_loader("/home/udorji/Documents/TRACK-3/generative-ai-assignment-ugyen505dorji-1686737015613/assets/Document.docx")
    # chunck_text = document_splitter(pages=pages)
    # vector_db = document_embedding(chunck_text=chunck_text)
    # print(summarize(vector_db=vector_db))
