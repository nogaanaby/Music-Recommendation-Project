import pandas as pd
from sklearn.preprocessing import OneHotEncoder

artist_tag = pd.read_csv("datasets/small_DTS/artist_tag_small.csv")
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded_tags = encoder.fit_transform(artist_tag[["tag"]])
encoded_df = pd.DataFrame(encoded_tags, columns=encoder.categories_[0])
artist_one_hot = pd.concat([artist_tag[["artist_id"]], encoded_df], axis=1)

# Remove duplicate artist rows by grouping and taking max (1 if artist has the tag, else 0)
artist_one_hot = artist_one_hot.groupby("artist_id").max().reset_index()

artist_one_hot.to_csv("new_small_DTS/artist_tags_one_hot.csv", index=False)
print(artist_one_hot.head())
