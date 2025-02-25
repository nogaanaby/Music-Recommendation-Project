import pandas as pd
user_interactions = pd.read_csv("datasets/small_DTS/active_users_interactions.csv")
# Aggregate play counts per user-artist pair
user_artist_interaction = user_interactions.groupby(["user_id", "artist_id"]).agg({"play_count": "sum"}).reset_index()
# Add 'like' column based on play count conditions
user_artist_interaction["like"] = user_artist_interaction["play_count"].apply(lambda x: 1 if x > 2 else (0 if x <= 2 else None))
user_artist_interaction = user_artist_interaction.dropna(subset=["like"])

user_artist_interaction["like"] = user_artist_interaction["like"].astype(int)
user_artist_interaction.to_csv("datasets/processed_tables/user_like_artist.csv", index=False)
print(user_artist_interaction.head())
