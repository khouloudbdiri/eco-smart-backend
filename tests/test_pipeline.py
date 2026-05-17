"""
Tests automatisés — Eco-Smart Classifier
🔴 Zone rouge : écrit manuellement sans IA

Chaque fonction test_ vérifie une chose précise.
Si elle lève une AssertionError → le test échoue ❌
Si elle se termine normalement → le test passe ✅

Pour lancer : pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ajouter le dossier racine au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════
# TEST 1 : Schéma des données
# Vérifie que toutes les colonnes obligatoires existent
# ════════════════════════════════════════════
def test_schema_train():
    """Le fichier train.csv doit contenir toutes les colonnes nécessaires."""
    df = pd.read_csv('./data/processed/train.csv')

    colonnes_obligatoires = [
        'Poids', 'Volume', 'Conductivite',
        'Opacite', 'Rigidite', 'Source_enc',
        'Categorie_enc', 'Prix_Revente', 'Categorie'
    ]

    for col in colonnes_obligatoires:
        assert col in df.columns, f"❌ Colonne manquante dans train.csv : {col}"

    print(f"✅ Schéma OK — {len(df.columns)} colonnes trouvées")


# ════════════════════════════════════════════
# TEST 2 : Qualité post-imputation
# Vérifie qu'il n'y a plus de NaN dans les colonnes numériques
# ════════════════════════════════════════════
def test_pas_de_nan_apres_imputation():
    """Après nettoyage, les colonnes numériques ne doivent pas avoir de NaN."""
    df = pd.read_csv('./data/processed/train.csv')

    cols_numeriques = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    nb_nan = df[cols_numeriques].isna().sum().sum()

    assert nb_nan == 0, f"❌ {nb_nan} valeurs NaN trouvées après imputation"
    print("✅ Pas de NaN dans les colonnes numériques")


# ════════════════════════════════════════════
# TEST 3 : Taille du split
# Vérifie que le ratio 70/15/15 est respecté
# ════════════════════════════════════════════
def test_taille_splits():
    """Les splits doivent respecter approximativement le ratio 70/15/15."""
    df_train = pd.read_csv('./data/processed/train.csv')
    df_val   = pd.read_csv('./data/processed/val.csv')
    df_test  = pd.read_csv('./data/processed/test.csv')

    total = len(df_train) + len(df_val) + len(df_test)

    ratio_train = len(df_train) / total
    ratio_val   = len(df_val)   / total
    ratio_test  = len(df_test)  / total

    # Tolérance de 5%
    assert 0.65 <= ratio_train <= 0.75, f"❌ Ratio train incorrect : {ratio_train:.2f}"
    assert 0.10 <= ratio_val   <= 0.20, f"❌ Ratio val incorrect : {ratio_val:.2f}"
    assert 0.10 <= ratio_test  <= 0.20, f"❌ Ratio test incorrect : {ratio_test:.2f}"

    print(f"✅ Splits OK — Train={ratio_train:.0%} Val={ratio_val:.0%} Test={ratio_test:.0%}")


# ════════════════════════════════════════════
# TEST 4 : Fichiers pkl existent
# Vérifie que tous les modèles ont bien été sauvegardés
# ════════════════════════════════════════════
def test_fichiers_pkl_existent():
    """Tous les fichiers .pkl nécessaires doivent exister."""
    fichiers_requis = [
        './data/processed/best_classifier.pkl',
        './data/processed/scaler.pkl',
        './data/processed/label_encoder.pkl',
        './data/processed/tfidf_vectorizer.pkl',
        './data/processed/kmeans_model.pkl',
        './data/processed/pca_model.pkl',
    ]

    for fichier in fichiers_requis:
        assert os.path.exists(fichier), f"❌ Fichier manquant : {fichier}"

    print(f"✅ {len(fichiers_requis)} fichiers pkl trouvés")


# ════════════════════════════════════════════
# TEST 5 : Performance minimale du classifieur
# Le modèle doit avoir accuracy >= 0.70
# ════════════════════════════════════════════
def test_performance_minimale():
    """Le meilleur classifieur doit avoir accuracy >= 0.70 sur le test set."""
    from sklearn.metrics import accuracy_score

    model  = joblib.load('./data/processed/best_classifier.pkl')
    df_te  = pd.read_csv('./data/processed/test.csv')

    FEATURES = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Source_enc']
    X_test   = df_te[FEATURES]
    y_test   = df_te['Categorie_enc']

    acc = accuracy_score(y_test, model.predict(X_test))

    assert acc >= 0.70, f"❌ Accuracy trop basse : {acc:.4f} < 0.70"
    print(f"✅ Accuracy OK : {acc:.4f} >= 0.70")


# ════════════════════════════════════════════
# TEST 6 : Pipeline NLP — vectorisation
# Vérifie que le TF-IDF produit le bon format
# ════════════════════════════════════════════
def test_pipeline_nlp_vectorisation():
    """Le vectoriseur TF-IDF doit produire 500 features."""
    tfidf = joblib.load('./data/processed/tfidf_vectorizer.pkl')

    texte_test = ["déchet métallique conducteur rigide très lourd"]
    vec = tfidf.transform(texte_test)

    assert vec.shape[1] == 500, f"❌ TF-IDF doit avoir 500 features, a {vec.shape[1]}"
    assert vec.nnz > 0,         "❌ Le vecteur TF-IDF est entièrement vide"
    print(f"✅ TF-IDF OK — shape={vec.shape}, valeurs non-nulles={vec.nnz}")


# ════════════════════════════════════════════
# TEST 7 : Prédiction NLP cohérente
# Vérifie que le pipeline NLP prédit une classe valide
# ════════════════════════════════════════════
def test_prediction_nlp_valide():
    """Le pipeline NLP doit prédire une classe parmi les 4 catégories."""
    tfidf  = joblib.load('./data/processed/tfidf_vectorizer.pkl')
    clf    = joblib.load('./data/processed/nlp_best_classifier.pkl')
    le     = joblib.load('./data/processed/label_encoder.pkl')

    texte_clean = "metal conducteur rigide lour"
    vec         = tfidf.transform([texte_clean])
    pred_enc    = clf.predict(vec)[0]
    pred_label  = le.inverse_transform([pred_enc])[0]

    classes_valides = list(le.classes_)
    assert pred_label in classes_valides, \
        f"❌ Prédiction invalide : '{pred_label}' pas dans {classes_valides}"
    print(f"✅ NLP prédit : '{pred_label}' — classes valides : {classes_valides}")


# ════════════════════════════════════════════
# TEST 8 : API endpoint /predict/numerique
# Vérifie que l'API répond correctement
# ════════════════════════════════════════════
def test_api_predict_numerique():
    """L'endpoint /predict/numerique doit retourner une catégorie valide."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    payload = {
        "Poids":        1.5,
        "Volume":       0.3,
        "Conductivite": 0.8,
        "Opacite":      0.2,
        "Rigidite":     0.9,
        "Source_enc":   1.0
    }

    response = client.post("/predict/numerique", json=payload)

    assert response.status_code == 200, \
        f"❌ Status code : {response.status_code} (attendu 200)"

    data = response.json()
    assert "categorie" in data, "❌ Champ 'categorie' manquant dans la réponse"
    assert data["categorie"] in ["Métal", "Papier", "Plastique", "Verre"], \
        f"❌ Catégorie invalide : {data['categorie']}"

    print(f"✅ API OK — catégorie prédite : {data['categorie']}")


# ════════════════════════════════════════════
# TEST 9 : API endpoint /predict/nlp
# ════════════════════════════════════════════
def test_api_predict_nlp():
    """L'endpoint /predict/nlp doit retourner une catégorie valide."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    payload = {"texte": "matériau métallique très conducteur et rigide"}

    response = client.post("/predict/nlp", json=payload)

    assert response.status_code == 200, \
        f"❌ Status code : {response.status_code}"

    data = response.json()
    assert "categorie" in data, "❌ Champ 'categorie' manquant"
    print(f"✅ API NLP OK — catégorie : {data['categorie']}")


# ════════════════════════════════════════════
# TEST 10 : API health check
# ════════════════════════════════════════════
def test_api_health():
    """L'endpoint /health doit retourner status=healthy."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ API health OK")