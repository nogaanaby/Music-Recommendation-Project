import pandas as pd
import os

user_like_artist = pd.read_csv("datasets/processed_tables/user_like_artist.csv", index_col=0)
artist_tags = pd.read_csv("datasets/processed_tables/artist_tags_pca.csv", index_col=0)
user_preference_matrix = pd.read_csv("datasets/processed_tables/user_preference_pca.csv", index_col=0)
artists = pd.read_csv("datasets/small_DTS/artists_small.csv", index_col=0)
users=pd.read_csv("datasets/small_DTS/active_500_users.csv", index_col=0)
train_artists = artists.index[:100]

output_file = "datasets/processed_tables/user_artist_RF_table.csv"
file_exists = False
last_user = user_like_artist.index[0]

if not os.path.exists(output_file):
    # Create an empty DataFrame with the desired columns
    empty_df = pd.DataFrame(columns=["user_id", "artist_id", "liked"] +
                                     user_preference_matrix.columns.tolist() +
                                     artist_tags.columns.tolist())
    empty_df.to_csv(output_file, index=False)
    file_exists = True
else:
    # Open the file and retrieve the last user
    try:
        existing_df = pd.read_csv(output_file)
        if not existing_df.empty:
            last_user = existing_df["user_id"].iloc[-1]
            print(f"Resuming from user: {last_user}")
        else:
            print("Existing file is empty.")
    except FileNotFoundError:
        print(f"File not found: {output_file}")
    except pd.errors.EmptyDataError:
        print("Existing file is empty.")

skip=True
for user in users.index:
    if user == last_user :
        skip = False
    if skip:
        continue

    current_user_data = user_like_artist.loc[user]
    his_artists = current_user_data["artist_id"]
    rows_to_append = []

    for artist in his_artists:
        if artist in train_artists:
            try:
                user_artist = current_user_data[current_user_data["artist_id"] == artist]
                like = user_artist["like"].values[0]
                row = {
                    "user_id": user,
                    "artist_id": artist,
                    "liked": like
                }
                row.update(user_preference_matrix.loc[user].to_dict())
                row.update(artist_tags.loc[artist].to_dict())
                rows_to_append.append(row)
            except KeyError:
                print(f"Warning: Artist '{artist}' not found in train_features.")

    if rows_to_append:
        df_chunk = pd.DataFrame(rows_to_append)
        df_chunk.to_csv(output_file, mode='a', header=not file_exists, index=False)
        print(f"Append data for user {user} \n")
        file_exists = True

print("Processing complete.")