import pandas as pd

user_artist_matrix = pd.read_csv("new_small_DTS/user_artist_encoded.csv", index_col=0)
artist_tags = pd.read_csv("new_small_DTS/artist_tags_one_hot.csv", index_col=0)


print("User-Artist Matrix Shape:", user_artist_matrix.shape)  # (num_users, num_artists)
print("Artist-Tags Matrix Shape:", artist_tags.shape)

user_preference_matrix = user_artist_matrix.dot(artist_tags)

user_preference_matrix.to_csv("new_small_DTS/user_preference.csv")

print(user_preference_matrix.head())
