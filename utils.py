def format_episode_id(season, episode):
    season = int(season)
    episode = int(episode)
    episode_id = "S{:02d}_EP{:02d}".format(season, episode)
    return episode_id

