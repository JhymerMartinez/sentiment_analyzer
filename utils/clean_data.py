import pandas as pd
import re
import string

BASE_DATASET_DIR = "data"
INPUT_DATASET_FILE_PATH = f"{BASE_DATASET_DIR}/training.1600000.processed.noemoticon.csv"
OUTPUT_DATASET_FILE_PATH = f"{BASE_DATASET_DIR}/cleaned_dataset.csv"

# Load the dataset
df = pd.read_csv(
    INPUT_DATASET_FILE_PATH,
    encoding="ISO-8859-1",
    header=None,
    names=["sentiment", "id", "date", "query", "user", "text"]
)

# Keep only the necessary columns
df = df[["sentiment", "text"]]

# Map sentiment classes
sentiment_map = {
    0: "negative",
    2: "neutral",
    4: "positive"
}

df["sentiment"] = df["sentiment"].map(sentiment_map)

# Rename the "sentiment" column to "label"
df.rename(columns={"sentiment": "label"}, inplace=True)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)             # Mentions
    text = re.sub(r"#\w+", "", text)             # Hashtags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Punctuation
    text = re.sub(r"\d+", "", text)              # Numbers
    text = re.sub(r"\s+", " ", text).strip()     # Extra spaces
    return text

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# Save cleaned file
df.to_csv(OUTPUT_DATASET_FILE_PATH, index=False)
