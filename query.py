import argparse
import pprint
# from dataclasses import dataclass
from langchain_chroma import Chroma 
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
# from langchain_core.runnables import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()
import os
from util import *
from cls_podcast_episode import *
import json
CHROMA_PATH = "chroma"


embedding_model = get_rag_embedding()

def load_sentence_transformer():
    # You can choose any pre-trained model from Sentence-Transformers
    return SentenceTransformer(embedding_model)  # or 'distilbert-base-nli-stsb-mean-tokens'
def vector_search(query_text, db, k):
    results = db.similarity_search_with_relevance_scores(query_text, k=6)
    return results
def get_vector_db():
    if embedding_model != "OPENAI":
        model = load_sentence_transformer()
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    else:
        embeddings = OpenAIEmbeddings()
    
    db = Chroma(
        embedding_function =embeddings,  # Pass HuggingFaceEmbeddings as 'embedding'
        persist_directory=CHROMA_PATH
    )
    return db
def get_model():
    return ChatOpenAI(model_name="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
def get_memory_handle():
    return ConversationBufferMemory(memory_key="history", return_messages=True)
def get_conversation_chain():
    return ConversationChain (
        llm=get_model(),
        memory=get_memory_handle(),
        verbose=False  # Set to True to view logs for debugging
    )
def build_prompt_with_context(query_text, search_results):
    if len(search_results) == 0 or search_results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in search_results])
    PROMPT_TEMPLATE = get_prompt_template("qa")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

def chat(db, conversation_chain):
    while True:
        # Ask the user for a question
        query_text = input("You: ")

        # Exit the loop if the user types 'quit'
        if query_text.lower() == "quit":
            print("Goodbye!")
            break

        # Search the DB.
        results = vector_search(query_text, db, 6)
        # setup prompt with context
        prompt = build_prompt_with_context(query_text, results)
        # get sources
        sources = get_sources(results)

        # Get the model's response based on the conversation chain
        response = conversation_chain.predict(input=prompt)
        formatted_response = f"Response: {response}\nSources: {sources}"
        # Print the response
        print("Bot:", formatted_response)

def main():
    db = get_vector_db()
    conversation_chain = get_conversation_chain()
    chat(db,conversation_chain)
    


if __name__ == "__main__":
    main()