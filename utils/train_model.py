import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from utils.clean_data import OUTPUT_DATASET_FILE_PATH

MODEL_BASE_DIR = "models"
MODEL_FILE_PATH = f"{MODEL_BASE_DIR}/sentiment_model.pkl"
METRICS_BASE_DIR = "metrics"
METRICS_FILE_PATH = f"{METRICS_BASE_DIR}/model_metrics.json"

# Load dataset
df = pd.read_csv(OUTPUT_DATASET_FILE_PATH)  # Asegúrate de que este archivo exista
df.dropna(subset=["text", "label"], inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Create and train the model
model = make_pipeline( TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Save the model
joblib.dump(model, MODEL_FILE_PATH)

# Save metrics
with open(METRICS_FILE_PATH, "w") as f:
    json.dump(report, f, indent=4)

print("Model and metrics saved successfully.")
