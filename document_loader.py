from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings

# from sentence_transformers import SentenceTransformer
# from langchain_chroma import Chroma 
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

import os
import time
from util import *
DATA_PATH = os.getenv("RAG_LOCAL_DATA_PATH")
embedding_model = get_rag_embedding()


def extract_metadata(json_data, _):
    return {
        "page_content": json_data.get("text", ""),  # Extract transcript text
        "episode_name": json_data.get("Episode Name", ""),
        "podcast_name": json_data.get("Podcast Name", ""),
        "episode_link": json_data.get("Episode Link", ""),
        "duration": json_data.get("duration", ""),
    }
def load_documents():
    log("info", f"Loading Documents  ")
    start_time = time.perf_counter() 
    
    # Use DirectoryLoader to load JSON transcripts
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",  # JSON files stored with .txt extension
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",  # Load entire JSON structure
            "text_content": False,  # Use "text" field as the main content
            "metadata_func": extract_metadata,
            },
        
    )
    documents = loader.load()
    for doc in documents:
        doc.page_content = doc.metadata.pop("page_content", "")  # Move text from metadata to content

    end_time = time.perf_counter() 
    elapsed_time = end_time - start_time 
    
    log("info", f"doc loading complete. Took {elapsed_time} seconds")
    return documents


def split_text(documents: list[Document]):
    log("info", f"*** Start Chunking ***")
    start_time = time.perf_counter() 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    end_time = time.perf_counter() 
    elapsed_time = end_time - start_time 
    log("info", f"Split {len(documents)} documents into {len(chunks)} chunks. Took {elapsed_time} seconds")

    return chunks

