import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load user interactions (label data)
user_data = pd.read_csv("small_DTS/user_like_artist_small.csv")

# Load artist metadata
artist_terms = pd.read_csv("small_DTS/artist_terms.csv")
artist_mbtag = pd.read_csv("small_DTS/artist_mbtag.csv")

# Merge data
df = user_data.merge(artist_terms, on="artist_id", how="left")
df = df.merge(artist_mbtag, on="artist_id", how="left")

# Fill missing values with empty strings
df.fillna("", inplace=True)




############################### One-Hot Encode Categorical Features ###############################
