import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

user_preference_matrix = pd.read_csv("datasets/processed_tables/user_preference_pca.csv", index_col=0)
user_like_artists = pd.read_csv("datasets/processed_tables/user_like_artist.csv", index_col=0)
artists = pd.read_csv("datasets/small_DTS/artists_small.csv", index_col=0)


############################################ METHODS ############################################

def pearson_similarity_matrix(df):
    similarity = np.corrcoef(df.values)
    return pd.DataFrame(similarity, index=df.index, columns=df.index)

def manhattan_distance_matrix(df):
    distances = pairwise_distances(df, metric='manhattan')
    return pd.DataFrame(distances, index=df.index, columns=df.index)

def cosine_similarity_matrix(df):
    similarity = cosine_similarity(df.values)
    return pd.DataFrame(similarity, index=df.index, columns=df.index)

############################################ plot ############################################

def plot_similarity_matrix(sim_matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

############################################ Evaluation ############################################

def get_user_artist_ids(user_id):
    user=user_like_artists.loc[user_id]
    user_artists_listend = None
    user_artists_liked = None
    if isinstance(user, pd.Series):
        user_artists_listend = {user["artist_id"]}
        user_artists_liked = {user["artist_id"]} if user["like"] == 1 else set()
    else:
        user_artists_listend = set(user["artist_id"])
        user_artists_liked = set(user.loc[user["like"] == 1, "artist_id"])

    return user_artists_listend, user_artists_liked

def similar_users_accuracy(user_id1, user_id2):
    user1_artists, _ = get_user_artist_ids(user_id1)
    user2_artists, _ = get_user_artist_ids(user_id2)

    common_artists = user1_artists.intersection(user2_artists)
    all_artists = user1_artists.union(user2_artists)
    if len(all_artists) == 0:
        return 0
    acc = (len(common_artists) * 100) / len(all_artists)
    return acc

def evaluate_by_closest_user(sim_matrix):
    total_acc = 0
    pairs = []
    for user in sim_matrix.index:
        similarities = sim_matrix.loc[user].copy()
        # Exclude self-comparison by setting its similarity to a very small number
        similarities[user] = -np.inf
        # Identify the most similar user
        closest_user = similarities.idxmax()
        pairs.append((user, closest_user))

        total_acc += similar_users_accuracy(user, closest_user)

    average_acc = total_acc / len(sim_matrix.index)
    return average_acc, pairs

############################################ Compare ############################################

### Cosine ####
cosine_sim_matrix = cosine_similarity_matrix(user_preference_matrix)
plot_similarity_matrix(cosine_sim_matrix, 'Users Cosine Similarity')
cos_avg_accuracy, cos_similar_pairs = evaluate_by_closest_user(cosine_sim_matrix)

print("Cosine Similarity - Average accuracy by closest user:", cos_avg_accuracy)

### Pearson ####
pearson_sim_matrix = pearson_similarity_matrix(user_preference_matrix)
plot_similarity_matrix(pearson_sim_matrix, 'Users Pearson Correlation')
pear_avg_accuracy, pear_similar_pairs = evaluate_by_closest_user(pearson_sim_matrix)

print("Pearson Correlation - Average accuracy by closest user:", pear_avg_accuracy)

### Manhattan ###
manhattan_sim_matrix = manhattan_distance_matrix(user_preference_matrix)
plot_similarity_matrix(manhattan_sim_matrix, 'Users Manhattan Distance')
manh_avg_accuracy, manh_similar_pairs = evaluate_by_closest_user(manhattan_sim_matrix)

print("Manhattan Distance - Average accuracy by closest user:", manh_avg_accuracy)

############################################ Recommendation ############################################
def recommend_artists(user_id, similarity_matrix):
    user = user_like_artists.loc[user_id]
    if isinstance(user, pd.Series):
        # If only one record exists for the user
        artists_the_user_know = {user["artist_id"]} if user["like"] == 1 else set()
    else:
        # If the user has multiple entries, filter for liked artists
        artists_the_user_know = set(user.loc[user["like"] == 1, "artist_id"])

    # All available artists, assuming the index of `artists` is the artist_id.
    all_artists = set(artists.index)

    # Artists that the user does NOT know.
    artists_the_user_dont_know = all_artists - artists_the_user_know

    # Find the most similar user (excluding self).
    similarities = similarity_matrix.loc[user_id].copy()
    similarities[user_id] = -np.inf  # Exclude self-comparison.
    similar_user = similarities.idxmax()

    # Retrieve the similar user's liked artists (only those with like==1)
    similar_user_data = user_like_artists.loc[similar_user]
    if isinstance(similar_user_data, pd.Series):
        similar_user_artists = {similar_user_data["artist_id"]} if similar_user_data["like"] == 1 else set()
    else:
        similar_user_artists = set(similar_user_data.loc[similar_user_data["like"] == 1, "artist_id"])

    # The recommended artists are those that the similar user likes that the current user hasn't seen.
    recommendation_set = similar_user_artists.intersection(artists_the_user_dont_know)

    return recommendation_set


# Recommand to specific user and add the artists names
u_id='fed07f4def346a276a553f7b831382f7acfca995'
recommend_artist_ids=recommend_artists(u_id,cosine_sim_matrix)
for artist_id in recommend_artist_ids:
    artist = artists.loc[artist_id]
    artist_name = artist["artist_name"]
    if len(artist_name) > 0:
        print(f"User '{u_id}' might like '{artist_name}'.")