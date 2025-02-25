import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
df = pd.read_csv("datasets/processed_tables/user_artist_RF_table.csv")

# Separate features (X) and target (y)
X = df.drop(["user_id", "artist_id", "liked"], axis=1)  # Drop non-feature columns
y = df["liked"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust n_estimators as needed
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)


model_filename = "artist_recommendation_model.joblib"
joblib.dump(rf_classifier, model_filename)
print(f"Model saved to {model_filename}")


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))
