import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load the data (take only 10,000 entries for efficiency)
songs_df = pd.read_csv('datasets/small_DTS/active_user_like_songs.csv')  # Columns: user_id, song_id, play_count, artist_id
artists_df = pd.read_csv('datasets/small_DTS/active_user_like_artist.csv')  # Columns: user_id, artist_id,total_play_count

# Binarize play_count (threshold = 5)
songs_df['like'] = (songs_df['play_count'] > 5).astype(int)

#redundant
# Aggregate duplicate (user_id, song_id) pairs by taking the max
songs_df = songs_df.groupby(['user_id', 'song_id'], as_index=False)['like'].max()

# Add a 'like' column to artists_df if the user heard him at least
artists_df['like'] = (artists_df['total_play_count'] > 15).astype(int)

# Aggregate duplicate (user_id, artist_id) pairs by taking the max
artists_df = artists_df.groupby(['user_id', 'artist_id'], as_index=False)['like'].max()

# Convert to user-item matrices
user_song_matrix = songs_df.pivot(index='user_id', columns='song_id', values='like').fillna(0)
user_artist_matrix = artists_df.pivot(index='user_id', columns='artist_id', values='like').fillna(0)

# Ensure train-test split maintains user overlap
common_users = list(set(user_song_matrix.index) & set(user_artist_matrix.index))  # Users present in both matrices
train_users, test_users = train_test_split(common_users, test_size=0.2, random_state=42)

train_songs = user_song_matrix.loc[train_users]
test_songs = user_song_matrix.loc[test_users]

train_artists = user_artist_matrix.loc[train_users]
test_artists = user_artist_matrix.loc[test_users]

# Compute similarity matrices
def compute_similarities(matrix):
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], matrix.shape[0])), np.zeros((matrix.shape[0], matrix.shape[0]))
    cosine_sim = cosine_similarity(matrix)
    pearson_sim = np.corrcoef(matrix)
    pearson_sim = np.nan_to_num(pearson_sim)  # Handle NaN values
    return cosine_sim, pearson_sim

cosine_sim_songs, pearson_sim_songs = compute_similarities(train_songs)
cosine_sim_artists, pearson_sim_artists = compute_similarities(train_artists)

# Plot Similarity Heatmaps
def plot_similarity_matrix(sim_matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

plot_similarity_matrix(cosine_sim_songs, 'Cosine Similarity - Songs')
plot_similarity_matrix(cosine_sim_artists, 'Cosine Similarity - Artists')

# KNN Model
def knn_recommend(matrix, k=5):
    if matrix.shape[0] < k:
        k = matrix.shape[0]
    model = NearestNeighbors(n_neighbors=k, metric='cosine')
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    return distances, indices

knn_songs_dist, knn_songs_indices = knn_recommend(train_songs)
knn_artists_dist, knn_artists_indices = knn_recommend(train_artists,15)

# Recommendation Function
def recommend(user_id, similarity_matrix, matrix):
    if user_id not in matrix.index:
        return []  # Skip users not in the training set
    user_index = matrix.index.get_loc(user_id)
    similar_users = np.argsort(-similarity_matrix[user_index])[:5]
    recommended_items = []
    for u in similar_users:
        recommended_items.extend(matrix.iloc[u].index[matrix.iloc[u] > 0].tolist())
    return list(set(recommended_items))

# Evaluate Recommendation Accuracy
def evaluate_recommendations(test_matrix, similarity_matrix, matrix):
    total_correct = 0
    total_hidden = 0
    for user_id in test_matrix.index:
        if user_id not in matrix.index:
            continue  # Skip evaluation for users not in the training set
        hidden_items = test_matrix.loc[user_id][test_matrix.loc[user_id] > 0].index
        recommended = recommend(user_id, similarity_matrix, matrix)
        correct = len(set(recommended) & set(hidden_items))
        total_correct += correct
        total_hidden += len(hidden_items)
    return total_correct / total_hidden if total_hidden > 0 else 0

song_accuracy = evaluate_recommendations(test_songs, cosine_sim_songs, train_songs)
artist_accuracy = evaluate_recommendations(test_artists, cosine_sim_artists, train_artists)

print(f'Song Recommendation Accuracy: {song_accuracy:.2f}')
print(f'Artist Recommendation Accuracy: {artist_accuracy:.2f}')
