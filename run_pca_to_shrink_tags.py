import pandas as pd
from sklearn.decomposition import PCA

artist_tags = pd.read_csv("datasets/processed_tables/artist_tags_one_hot.csv", index_col=0)
user_preference_matrix = pd.read_csv("datasets/processed_tables/user_preference.csv", index_col=0)

pca_artist = PCA(n_components=0.95)
artist_tags_pca = pca_artist.fit_transform(artist_tags)
artist_tags_pca_df = pd.DataFrame(artist_tags_pca, index=artist_tags.index)
artist_tags_pca_df.columns = [f"artist_pca_{i}" for i in range(artist_tags_pca_df.shape[1])]

pca_user = PCA(n_components=0.95)
user_preference_pca = pca_user.fit_transform(user_preference_matrix)
user_preference_pca_df = pd.DataFrame(user_preference_pca, index=user_preference_matrix.index)
user_preference_pca_df.columns = [f"user_pca_{i}" for i in range(user_preference_pca_df.shape[1])]

artist_tags_pca_df.to_csv("datasets/processed_tables/artist_tags_pca.csv")
user_preference_pca_df.to_csv("datasets/processed_tables/user_preference_pca.csv")

