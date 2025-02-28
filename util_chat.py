# from dataclasses import dataclass
from langchain_chroma import Chroma 
from sentence_transformers import SentenceTransformer, util
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseMessage
# from langchain.memory import ConversationBuffer

from rapidfuzz import process
from dotenv import load_dotenv
load_dotenv()
import os
from util import *
from cls_podcast_episode import *
from summarizer import *
import json
CHROMA_PATH = get_vector_db_location()


embedding_model = get_rag_embedding()


def load_sentence_transformer():
    # You can choose any pre-trained model from Sentence-Transformers
    return SentenceTransformer(embedding_model)  # or 'distilbert-base-nli-stsb-mean-tokens'
def vector_search(query_text, db, k=6):
    results = db.similarity_search_with_relevance_scores(query_text, k)
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
    # return ConversationBuffer(memory_key="history", return_messages=True)
    

# decision making
def process_user_input(user_input):
    # Simple intent check based on keywords
    if any(keyword in user_input.lower() for keyword in ['summary', 'summarize', 'overview']):
        return 'summary'
    else:
        return 'question'


class SessionHistory:
    def __init__(self, messages):
        self.messages = messages  # Ensure the object has a `.messages` attribute
    def add_messages(self, new_messages):
        """LangChain expects an `add_messages` method, so we append to the list."""
        self.messages.extend(new_messages)

def get_conversation_chain():
    session_id = "XXX"
    return RunnableWithMessageHistory(
        runnable=get_model(),
        get_session_history=lambda session_id: SessionHistory(
            get_memory_handle().load_memory_variables(session_id).get("history", [])
        ),
        verbose=False 
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
    session_id = "XXX"
    while True:
        # Ask the user for a question
        query_text = input("You: ")
        

        # Exit the loop if the user types 'quit'
        if query_text.lower() == "quit":
            print("Goodbye!")
            break

        intent = process_user_input(query_text)

        # Search the DB.
        results = vector_search(query_text, db, 6)
        # setup prompt with context
        prompt = build_prompt_with_context(query_text, results)
        
        # get sources
        sources = get_sources(results)
        if intent == 'summary': # Summary - use local ollama
            content = get_source_episode_context(results)
            # Summarize the episodes based on the retrieved context
            response_text = create_summary(content)
        else: # normal Q&A - use openai
            # Get the model's response based on the conversation chain
            response = conversation_chain.invoke(
                input=prompt,
                config={"configurable": {"session_id": session_id}}
                )
            # Extract only the response text
            if isinstance(response, BaseMessage):
                # Extract message content from BaseMessage object
                response_text = response.content
            else:
                # If it's not a BaseMessage, fallback to string conversion
                response_text = str(response)
        # add sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        # Print the response
        print("Bot:", formatted_response)
