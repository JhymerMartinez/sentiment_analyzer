from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Sample training data
texts = [
    "me encanta este producto",
    "es horrible",
    "está bien",
    "muy malo",
    "lo recomiendo",
    "no me gustó",
    "servicio pésimo",
    "excelente calidad",
    "más o menos",
    "muy buen soporte",
    "me fascina este lugar",
    "altamente recomendable",
    "un excelente servicio",
    "quedé muy satisfecho",
    "la mejor experiencia",
    "definitivamente lo volveré a comprar",
    "estuvo bien, pero podría mejorar",
    "muy buena opción",
    "bastante bueno",
    "no lo recomendaría",
    "no es lo que esperaba",
    "un servicio terrible",
    "me decepcionó mucho",
    "estoy feliz con mi compra",
    "mejor que lo que imaginaba",
    "es una opción aceptable",
    "decepcionante",
    "gran calidad en el producto"
]

labels = [
    "positivo",
    "negativo",
    "neutro",
    "negativo",
    "positivo",
    "negativo",
    "negativo",
    "positivo",
    "neutro",
    "positivo",
    "positivo",
    "positivo",
    "positivo",
    "positivo",
    "positivo",
    "positivo",
    "neutro",
    "positivo",
    "positivo",
    "negativo",
    "negativo",
    "negativo",
    "negativo",
    "positivo",
    "positivo",
    "neutro",
    "negativo",
    "positivo"
]

# Train and save the model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)
joblib.dump(model, "sentiment_model.pkl")
