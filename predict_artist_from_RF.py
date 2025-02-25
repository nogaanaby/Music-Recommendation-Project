import joblib

model_filename = "artist_recommendation_model.joblib"
loaded_model = joblib.load(model_filename)

# Example: Predicting for a specific user and artist (replace with actual user and artist IDs)
user_id_to_predict = "YOUR_USER_ID"  # Replace with a valid user ID
artist_id_to_predict = "YOUR_ARTIST_ID"  # Replace with a valid artist ID

# Create a DataFrame with the user and artist data for prediction
user_data = df[df["user_id"] == user_id_to_predict].iloc[0].drop(["user_id", "artist_id", "liked"])  # Get user features
artist_data = df[df["artist_id"] == artist_id_to_predict].iloc[0].drop(["user_id", "artist_id", "liked"])  # Get artist features

# Combine the features
prediction_data = pd.concat([user_data, artist_data], axis=0).values.reshape(1, -1)

# Make the prediction
prediction = rf_classifier.predict(prediction_data)

print(f"Prediction for user {user_id_to_predict} and artist {artist_id_to_predict}: {prediction[0]}")