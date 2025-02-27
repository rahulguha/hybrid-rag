# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
# import openai 

import shutil
from operator import itemgetter

from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
import os
import time
from util import *
from document_loader import *
import json


# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()


CHROMA_PATH = get_vector_db_location()
DATA_PATH = os.getenv("RAG_LOCAL_DATA_PATH")
embedding_model = get_rag_embedding()

def download_corpus(DATA_PATH=os.getenv("RAG_LOCAL_DATA_PATH")):
    log("info", f"Downloading corpus to {DATA_PATH}  ")
    download_s3_bucket_flat( DATA_PATH, "transcriptions")   
    log("info", f"Downloading complete")
def load_sentence_transformer():
    # You can choose any pre-trained model from Sentence-Transformers
    return SentenceTransformer(embedding_model)  # or 'distilbert-base-nli-stsb-mean-tokens'


def build_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def save_to_chroma(chunks: list[Document]):
    log("info", f"*** Start Embedding  ***")
    start_time = time.perf_counter() 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.    
    if embedding_model != "OPENAI":
        model = load_sentence_transformer()
        embeddings = HuggingFaceEmbeddings(model_name=model)
    else:
        embeddings = OpenAIEmbeddings()
    

    # Persist
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    end_time = time.perf_counter() 
    elapsed_time = end_time - start_time 
    log("info", f"Saved {len(chunks)} chunks to {CHROMA_PATH}. Took {elapsed_time} seconds")

