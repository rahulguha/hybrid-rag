from util import *

import requests


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


def create_summary(content):
    
    system_prompt = '''
                    Summarize the text given to you in 3 bullet points roughly 200 words in length. 
                    Remove any product promotions.
                    Be specific, remove any guesses, use third person like "This episodes talks about ..."
                    Please don't use following word "trinscription", "appears to", "podcast", "episode" etc. in response.
                    Don't exceed the 200 word limit. Be formal about your style
                    '''

    OLLAMA_PROMPT = f"{system_prompt}: {content}"
    # print (OLLAMA_PROMPT)
    OLLAMA_DATA = {
        # "model": "llama3.2",
        # "model": "deepseek-r1",
        "model": "gemma2:latest",
        "prompt": OLLAMA_PROMPT,
        "stream": False,
        "keep_alive": "1m",
    }
    try : 
        response = requests.post(OLLAMA_ENDPOINT, json=OLLAMA_DATA)
    except Exception as e:
        print (e)
        return "Error in summarizing the content"
    return response.json()["response"]
            
    
