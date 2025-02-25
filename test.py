import pandas as pd

# Load the user-like artist matrix
user_like_matrix = pd.read_csv("datasets/processed_tables/user_like_artist.csv")

# Filter and print rows where like == 0
like_zero_rows = user_like_matrix[user_like_matrix["like"] == 0]
print(like_zero_rows)
