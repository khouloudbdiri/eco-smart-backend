FROM python:3.10-slim

WORKDIR /app

# Copier les dépendances en premier (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires
RUN python -m nltk.downloader punkt stopwords wordnet punkt_tab

# Copier tout le projet
COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]