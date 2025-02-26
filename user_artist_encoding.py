import pandas as pd

df = pd.read_csv("new_small_DTS/active_users_interactions.csv")

# Aggregate play counts per user-artist pair
df_artist = df.groupby(["user_id", "artist_id"])["play_count"].sum().reset_index()

# Calculate total play counts per user
user_total_play_counts = df_artist.groupby("user_id")["play_count"].sum().reset_index()
user_total_play_counts.rename(columns={"play_count": "total_play_counts"}, inplace=True)

# Merge total play counts back to the interaction data
df_artist = df_artist.merge(user_total_play_counts, on="user_id")

# Normalize play_count by dividing by total_play_counts
df_artist["normalized_play_count"] = df_artist["play_count"] / df_artist["total_play_counts"]

# Pivot the table to get user-artist matrix
user_artist_matrix = df_artist.pivot(index="user_id", columns="artist_id", values="normalized_play_count").fillna(0)

user_artist_matrix.to_csv("new_small_DTS/user_artist_encoded.csv")
print(user_artist_matrix.head())
