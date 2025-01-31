from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import mlflow.sklearn
from typing import List

# Initialisation de l'API
app = FastAPI(
    title="MLOps Demo API",
    description="API de prédiction avec FastAPI & Swagger UI 🚀",
    version="1.0"
)

# Chargement du modèle ML
try:
    model = joblib.load("random_forest.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Could not load model: {e}")

# Définition d'un modèle de requête avec des valeurs par défaut
class InputData(BaseModel):
    features: List[float] = Field(
        default=[5.1, 3.5, 1.4, 0.2], 
        example=[5.1, 3.5, 1.4, 0.2],
        description="Liste de valeurs numériques pour la prédiction"
    )

# Route d'accueil
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction MLOps 🚀", "docs": "/docs"}

# Route de prédiction
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not loaded"}

    prediction = model.predict([data.features])[0]
    return {
        "features": data.features,
        "prediction": int(prediction),
        "model": "RandomForestClassifier",
    }

# Exécution en local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
