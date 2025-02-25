import pandas as pd

user_like_artist = pd.read_csv("datasets/processed_tables/user_like_artist.csv", index_col=0)
artist_tags = pd.read_csv("datasets/processed_tables/artist_tags_pca.csv", index_col=0)
user_preference_matrix = pd.read_csv("datasets/processed_tables/user_preference_pca.csv", index_col=0)
artists= pd.read_csv("datasets/small_DTS/artists_small.csv", index_col=0)

# # Ensure artist IDs match
# artist_tags = artist_tags.reindex(user_like_artist.columns)
# artist_tags = artist_tags.dropna()

# Step 1: Split the Data (Train: First 100 artists, Test: Remaining artists)
train_artists = artists.index[:100]
test_artists = artists.index[100:]

# Step 4: Create Features for train Artists
train_features = artist_tags.loc[train_artists]

# Step 3: Combine Everything into Final Table
final_df = []


for user in user_like_artist.index:
    current_user_data = user_like_artist.loc[user]
    his_artists=current_user_data["artist_id"]
    for artist in his_artists:
        # Check if the artist exists in user_like_artist
        user_artist = current_user_data[current_user_data["artist_id"]==artist]
        like=user_artist["like"].values[0]

        if artist in train_artists:
            row = {
                "user_id": user,
                "artist_id": artist,
                "liked": like
            }

            row.update(user_preference_matrix.loc[user].to_dict())
            row.update(train_features.loc[artist].to_dict())
            final_df.append(row)

# Convert the list of rows into a DataFrame
final_df = pd.DataFrame(final_df)

# Save the final table to a CSV file
final_df.to_csv("datasets/processed_tables/user_artist_RF_table.csv", index=False)

# Print sample output
print(final_df.head())