import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("datasets/processed_tables/user_artist_RF_table.csv")

X = df.drop(["user_id", "artist_id", "liked"], axis=1)  # Drop non-feature columns
y = df["liked"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)


model_filename = "RF_data/artist_recommendation_model.joblib"
joblib.dump(rf_classifier, model_filename)
print(f"Model saved to {model_filename}")


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

X_test.to_csv("RF_data/X_test.csv", index=False)
y_test.to_csv("RF_data/y_test.csv", index=False)
