
from util import *
from util_chat import *
import json

def main():
    db = get_vector_db()
    conversation_chain = get_conversation_chain()
    chat(db,conversation_chain)
    


if __name__ == "__main__":
    main()