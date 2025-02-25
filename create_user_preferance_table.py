import pandas as pd

# Load user-artist interaction matrix
user_artist_matrix = pd.read_csv("new_small_DTS/user_artist_encoded.csv", index_col=0)

# Load artist-tag matrix (one-hot encoded tags for artists)
artist_tags = pd.read_csv("new_small_DTS/artist_tags_one_hot.csv", index_col=0)


print("User-Artist Matrix Shape:", user_artist_matrix.shape)  # (num_users, num_artists)
print("Artist-Tags Matrix Shape:", artist_tags.shape)

# Compute user preferences as a weighted sum of artist tags
user_preference_matrix = user_artist_matrix.dot(artist_tags)

# Save the result to a CSV file
user_preference_matrix.to_csv("new_small_DTS/user_preference.csv")

# Print sample output
print(user_preference_matrix.head())
