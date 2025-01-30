FROM python:3.10.10-slim

# 1) Définir le répertoire de travail
WORKDIR /app

# 2) Copier Pipfile & Pipfile.lock
COPY Pipfile Pipfile.lock ./

# 3) Installer Pipenv et les dépendances système
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --deploy

# 4) Copier le code source dans /app/src
COPY src/ ./src
COPY random_forest.pkl .


# 5) Exposer le port 5000 (optionnel mais pratique)
EXPOSE 5000

# 6) Commande de lancement
CMD ["python", "src/api.py"]
