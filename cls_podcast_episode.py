class PodcastEpisode:
    def __init__(self, episode_name: str, episode_link: str, podcast_name: str):
        self.episode_name = episode_name
        self.episode_link = episode_link
        self.podcast_name = podcast_name 

    def __repr__(self):
        return f"PodcastEpisode(episode_name='{self.episode_name}', episode_link='{self.episode_link}', podcast_name='{self.podcast_name}')"

    
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