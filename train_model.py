import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/symptoms_dataset.csv")
X = df.drop("disease", axis=1)
y = df["disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and feature list
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(list(X.columns), "model/symptom_list.pkl")

print("✅ Model trained and saved.")
