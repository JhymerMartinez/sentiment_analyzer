# Sentiment Analyzer API

This is a sentiment analysis API built with Flask. It uses a logistic regression model trained on a cleaned version of the Sentiment140 dataset to classify Spanish input text as `positive`, `negative`, or `neutral`. The API also provides actionable recommendations based on the sentiment.

---

## Features

- Clean and prepare raw tweets using `utils/clean_data.py`
- Train a logistic regression model with TF-IDF vectorization
- Predict sentiment using a `/predict` endpoint
- Access model evaluation metrics via a `/metrics` endpoint
- Modular structure with scripts and models organized in folders
- Ready for local development with `pyenv`

---

## Prerequisites

- Python 3.10.13 (recommended via `pyenv`)
- `pip` package manager

If you're using `pyenv`, the project includes a `.python-version` file:

```bash
pyenv install 3.10.13  # if not already installed
pyenv local 3.10.13
```

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/JhymerMartinez/sentiment_analyzer.git
cd sentiment_analyzer
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Prepare the Dataset

1. Download the original Sentiment140 dataset from [here](https://www.kaggle.com/datasets/kazanova/sentiment140/data).

   - File: `training.1600000.processed.noemoticon.csv`
   - Place it inside the `data/` folder.

2. Run the data cleaning script:

```bash
python utils/clean_data.py
```

This will output a `cleaned_dataset.csv` file to the `data/` directory.

---

## Train the Model

Run the training script to train and save the model along with evaluation metrics:

```bash
python utils/train_model.py
```

This will generate:

- `models/sentiment_model.pkl` → trained model
- `metrics/model_metrics.json` → evaluation metrics

---

## Run the API

Start the Flask server:

```bash
python app.py
```

The API will be available at:
http://127.0.0.1:5000

---

## API Endpoints

### POST /predict

Send text input and get back a sentiment prediction with a recommendation.

**Example:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I like this product"}'
```

**Response:**

```json
{
  "sentiment": "positive",
  "recommendation": "Keep reinforcing what you're doing well."
}
```

---

### GET /metrics

Fetch the stored model evaluation metrics:

```bash
curl http://127.0.0.1:5000/metrics
```

**Example Response:**

```json
{
  "accuracy": 0.797,
  "macro avg": {
    "precision": 0.797,
    "recall": 0.797,
    "f1-score": 0.797
  },
  "weighted avg": {
    "precision": 0.797,
    "recall": 0.797,
    "f1-score": 0.797
  }
}
```

---

## Project Structure

```
sentiment_analyzer/
│
├── app.py
├── requirements.txt
├── .python-version
│
├── utils/
│   ├── clean_data.py
│   └── train_model.py
│
├── data/
│   ├── training.1600000.processed.noemoticon.csv
│   └── cleaned_dataset.csv
│
├── models/
│   └── sentiment_model.pkl
│
└── metrics/
    └── model_metrics.json
```

---

## Notes

- Make sure `data/`, `models/`, and `metrics/` directories exist before running the scripts.
- The files `sentiment_model.pkl`, `model_metrics.json`, and `cleaned_dataset.csv` are excluded from version control (see `.gitignore`).

---

## License

MIT License
