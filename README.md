# Sentiment Analyzer API

This is a simple sentiment analysis API built with Flask. It uses a logistic regression model trained on sample data to classify input text as `positive`, `negative`, or `neutral` and provides actionable recommendations based on the sentiment.

## Features

- Train and save a sentiment analysis model using `scikit-learn`.
- Expose a `/predict` endpoint with Flask for making predictions.
- Ready for local development using `pyenv`.

---

## Prerequisites

- Python 3.10.13 (recommended via `pyenv`)
- `pip` package manager

If you're using `pyenv`, the project includes a `.python-version` file to automatically select the Python version:

```bash
pyenv install 3.10.13  # if not already installed
pyenv local 3.10.13
```

---

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JhymerMartinez/sentiment_analyzer.git
   cd sentiment_analyzer
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Train the Model

Run the following script to train and save the model:

```bash
python train_model.py
```

This will generate a `sentiment_model.pkl` file that will be used by the Flask API.

---

## Run the API

Start the Flask app with:

```bash
python app.py
```

The API will be available at:
`http://127.0.0.1:5000/predict`

---

## Example Usage

Make a prediction using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "El servicio fue excelente"}'
```

**Response:**

```json
{
  "sentiment": "positivo",
  "recommendation": "Keep reinforcing what you're doing well."
}
```

---

## Notes

- The `.python-version` file ensures Python 3.10.13 is used with `pyenv`.
- The `sentiment_model.pkl` file is not tracked in version control (see `.gitignore`).

---

## License

MIT License
