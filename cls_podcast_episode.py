import json
import os

from dotenv import load_dotenv
load_dotenv()
from util import *
class PodcastEpisode:
    def __init__(self, episode_name: str, episode_link: str, podcast_name: str, filename: str=""):
        self.episode_name = episode_name
        self.episode_link = episode_link
        self.podcast_name = podcast_name 
        self.filename = filename 


    def __repr__(self):
        return f"PodcastEpisode(episode_name='{self.episode_name}', episode_link='{self.episode_link}', podcast_name='{self.podcast_name}, filename='{self.filename}')"

    
def get_sources(search_results):
    source_episodes = []
    episode_names_set = set()
    sources = [doc.metadata.get("metadata", None) for doc, _score in search_results]
    for s in search_results:
        episode_name = s[0].metadata['episode_name']
        if episode_name not in episode_names_set:
            episode = PodcastEpisode(
                episode_name=s[0].metadata['episode_name'],
                episode_link=s[0].metadata['episode_link'],
                podcast_name=s[0].metadata['podcast_name'],
            )
            source_episodes.append(episode)
            episode_names_set.add(episode_name)
    sources = ""
    sources = "\n\n".join([
        f"Podcast Name: {episode.podcast_name}\nEpisode Name: {episode.episode_name}\nEpisode Link: {episode.episode_link}"
        for episode in source_episodes
    ])
    return sources
def get_source_episode_context(search_results):
    source_episodes = []
    episode_names_set = set()
    # sources = [doc.metadata.get("metadata", None) for doc, _score in search_results]
    context = ""
    for s in search_results:
        episode_name = s[0].metadata['episode_name']
        if episode_name not in episode_names_set:
            # episode = PodcastEpisode(
            #     episode_name=s[0].metadata['episode_name'],
            #     episode_link=s[0].metadata['episode_link'],
            #     podcast_name=s[0].metadata['podcast_name'],
            # )
            context = context + get_transcript_for_episode(episode_name)

    
    return context

def get_transcript_for_episode(episode_name):
    transcript_location = get_local_rag_path()
    transcriptions = glob.glob(f"{get_local_rag_path()}/**/*.txt", recursive=True)  # Get all .txt files in subdirectories
    for i in range(len(transcriptions)):
      transcriptions[i] = f"{transcript_location}/" + strip_before_last_slash(transcriptions[i])
    context_text = ""
    # Load the transcript data from the JSON file
    for t in transcriptions:
        with open(t, 'r') as file:
            data = json.load(file)
            if data["Episode Name"] == episode_name:
                context_text = context_text + data["text"]
    return context_text



def get_all_transcription_episode_names():
    transcriptions = glob.glob(f"{get_local_rag_path()}/**/*.txt", recursive=True)  # Get all .txt files in subdirectories
    all_episodes=[]
    for i in range(len(transcriptions)):
    #   transcriptions[i] = strip_before_last_slash(transcriptions[i])
        
        with open(transcriptions[i], 'r') as file:
            data = json.load(file)
            episode = PodcastEpisode(
                episode_name=data["Episode Name"],
                episode_link=data["Episode Link"] ,
                podcast_name=data["Podcast Name"],
                filename=file.name
            )
        all_episodes.append(episode)
            # if data["Episode Name"] == episode_name:
            #     context_text = context_text + data["text"]
    
    return all_episodes



def get_transcript_from_file(filename):
    
    
    
    # filename = get_local_rag_path() + "/" + filename
    context_text = ""
    # Load the transcript data from the JSON file
    
    with open(filename, 'r') as file:
        data = json.load(file)
        context_text = context_text + data["text"]
    return context_text
        # Find the transcript for the specified episode
            # for episode in data:
            #     print (episode['Episode Name'])
                # if episode['Episode Name'] == episode_name:
                #     return episode['text']
    # return "Transcript not found for the specified episode."