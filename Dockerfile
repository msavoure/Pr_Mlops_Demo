# Utiliser une image Python légère
FROM python:3.10.10-slim

WORKDIR /app

# Copier les fichiers nécessaires
COPY Pipfile Pipfile.lock /app/
RUN pip install --no-cache-dir pipenv && pipenv install --system --deploy

COPY src/ /app/src/
COPY random_forest.pkl /app/

# Exposer le port utilisé par FastAPI
EXPOSE 5000

# Démarrer l’API avec Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "5000"]
