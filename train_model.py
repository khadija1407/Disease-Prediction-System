import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Read dataset
df = pd.read_csv("dataset.csv")

print(df.head())
print(df.columns)

# Features
X = df.drop(columns=["disease"])

# Target
y = df["disease"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "disease_model.pkl")

print("Model Trained Successfully!")