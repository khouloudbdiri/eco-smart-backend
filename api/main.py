from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI(title="Eco-Smart API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════
# CHARGEMENT DES MODÈLES
# ════════════════════════════════════════════
print("⏳ Chargement des modèles...")

le       = joblib.load('./data/processed/label_encoder.pkl')
scaler   = joblib.load('./data/processed/scaler.pkl')
best_clf = joblib.load('./data/processed/best_classifier.pkl')
tfidf    = joblib.load('./data/processed/tfidf_vectorizer.pkl')
nlp_clf  = joblib.load('./data/processed/nlp_best_classifier.pkl')
kmeans   = joblib.load('./data/processed/kmeans_model.pkl')
pca_model= joblib.load('./data/processed/pca_model.pkl')

FEATURES_NUM = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Source_enc']
COLS_SCALED  = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']

# ── Charger les stats réelles du dataset pour valider le scaler ──
df_train = pd.read_csv('./data/processed/train.csv')

# Vérifier que le scaler correspond aux données
print("=== Diagnostic scaler ===")
print(f"Scaler mean_  : {scaler.mean_}")
print(f"Scaler scale_ : {scaler.scale_}")
print(f"Train mean    : {df_train[COLS_SCALED].mean().values}")
print(f"Train std     : {df_train[COLS_SCALED].std().values}")

# ── Test de prédiction avec des valeurs typiques ──
print("\n=== Test prédiction avec valeurs typiques ===")
test_vals = df_train[FEATURES_NUM].iloc[0].values
print(f"Valeurs train[0] : {test_vals}")
pred_test = best_clf.predict([test_vals])
print(f"Prédiction : {le.inverse_transform(pred_test)}")

print("✅ Modèles chargés")

# ════════════════════════════════════════════
# PIPELINE MULTIMODAL
# ════════════════════════════════════════════
pipeline_multimodal = None
try:
    print("⏳ Construction du pipeline multimodal...")
    df_vl  = pd.read_csv('./data/processed/val.csv')
    df_fit = pd.concat([df_train, df_vl], ignore_index=True)

    if 'Rapport_clean' not in df_fit.columns:
        df_fit['Rapport_clean'] = ''

    X_fit = df_fit[FEATURES_NUM + ['Rapport_clean']]
    y_fit = df_fit['Categorie_enc']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), FEATURES_NUM),
            ('nlp', TfidfVectorizer(ngram_range=(1,2), max_features=500,
                                    min_df=2, sublinear_tf=True), 'Rapport_clean'),
        ],
        remainder='drop'
    )
    pipeline_multimodal = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier',   LinearSVC(max_iter=2000, random_state=42))
    ])
    pipeline_multimodal.fit(X_fit, y_fit)
    print("✅ Pipeline multimodal prêt")
except Exception as e:
    print(f"⚠️  Pipeline multimodal non disponible : {e}")

# ════════════════════════════════════════════
# NETTOYAGE TEXTE
# ════════════════════════════════════════════
STOPWORDS_FR     = set(stopwords.words('french'))
STOPWORDS_DOMAIN = {
    'collecte', 'rapport', 'materiau', 'dechet', 'lot',
    'site', 'usine', 'provenance', 'source', 'type'
}
STOPWORDS_ALL = STOPWORDS_FR | STOPWORDS_DOMAIN
lemmatizer    = WordNetLemmatizer()

def preprocess_text(texte: str) -> str:
    if not isinstance(texte, str) or texte.strip() == '':
        return ''
    texte  = texte.lower()
    texte  = re.sub(r'(\d+[\.,]?\d*)\s*(kg|g|cm|mm|l|ml|%)', r'\1\2', texte)
    texte  = re.sub(r'[^\w\s]', ' ', texte)
    tokens = word_tokenize(texte, language='french')
    tokens = [t for t in tokens if t not in STOPWORDS_ALL and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def prepare_input(poids, volume, conductivite, opacite, rigidite, source_enc):
    """
    Prépare le vecteur d'entrée pour le classifieur.
    
    IMPORTANT : Dans ton notebook, le scaler est fitté sur df AVANT le split.
    Les données dans train.csv sont DÉJÀ scalées.
    Donc le classifieur attend des valeurs SCALÉES.
    
    On utilise le scaler pour transformer les valeurs brutes des curseurs.
    """
    # Les 5 premières features sont scalées
    raw_5 = np.array([[poids, volume, conductivite, opacite, rigidite]])
    scaled_5 = scaler.transform(raw_5)[0]

    # Source_enc est une valeur ordinale (0, 1, 2...) — pas scalée
    return np.array([[
        scaled_5[0],  # Poids scalé
        scaled_5[1],  # Volume scalé
        scaled_5[2],  # Conductivite scalée
        scaled_5[3],  # Opacite scalée
        scaled_5[4],  # Rigidite scalée
        source_enc    # Source_enc non-scalée
    ]])


# ════════════════════════════════════════════
# MODÈLES DE DONNÉES
# ════════════════════════════════════════════

class PredictionNumerique(BaseModel):
    Poids:        float
    Volume:       float
    Conductivite: float
    Opacite:      float
    Rigidite:     float
    Source_enc:   float = 1.0

class PredictionNLP(BaseModel):
    texte: str

class PredictionMultimodale(BaseModel):
    Poids:        float
    Volume:       float
    Conductivite: float
    Opacite:      float
    Rigidite:     float
    Source_enc:   float = 1.0
    texte:        str   = ""


# ════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Eco-Smart API", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/numerique")
def predict_numerique(data: PredictionNumerique):
    X = prepare_input(
        data.Poids, data.Volume, data.Conductivite,
        data.Opacite, data.Rigidite, data.Source_enc
    )

    pred_enc   = best_clf.predict(X)[0]
    pred_label = le.inverse_transform([pred_enc])[0]

    try:
        proba     = best_clf.predict_proba(X)[0]
        confiance = float(round(proba.max() * 100, 1))
    except:
        confiance = None

    return {
        "categorie":     pred_label,
        "categorie_enc": int(pred_enc),
        "confiance":     confiance,
        "classes":       list(le.classes_),
    }


@app.post("/predict/nlp")
def predict_nlp(data: PredictionNLP):
    texte_clean = preprocess_text(data.texte)

    if texte_clean == '':
        return {
            "erreur":        "Texte trop court ou vide après nettoyage",
            "texte_nettoye": "",
            "categorie":     None,
            "classes":       list(le.classes_),
        }

    try:
        vec        = tfidf.transform([texte_clean])
        pred_enc   = nlp_clf.predict(vec)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        return {
            "categorie":     pred_label,
            "categorie_enc": int(pred_enc),
            "texte_nettoye": texte_clean,
            "classes":       list(le.classes_),
            "erreur":        None,
        }
    except Exception as e:
        return {
            "erreur":        f"Erreur : {str(e)}",
            "texte_nettoye": texte_clean,
            "categorie":     None,
            "classes":       list(le.classes_),
        }


@app.post("/predict/multimodal")
def predict_multimodal(data: PredictionMultimodale):
    if pipeline_multimodal is not None:
        df_input = pd.DataFrame([{
            'Poids':         data.Poids,
            'Volume':        data.Volume,
            'Conductivite':  data.Conductivite,
            'Opacite':       data.Opacite,
            'Rigidite':      data.Rigidite,
            'Source_enc':    data.Source_enc,
            'Rapport_clean': preprocess_text(data.texte),
        }])
        pred_enc = pipeline_multimodal.predict(df_input)[0]
    else:
        X        = prepare_input(
            data.Poids, data.Volume, data.Conductivite,
            data.Opacite, data.Rigidite, data.Source_enc
        )
        pred_enc = best_clf.predict(X)[0]

    pred_label = le.inverse_transform([pred_enc])[0]
    return {
        "categorie":     pred_label,
        "categorie_enc": int(pred_enc),
        "classes":       list(le.classes_),
    }


@app.get("/dashboard/data")
def get_dashboard_data():
    df_pca = pd.read_csv('./data/processed/pca_clusters.csv')
    return {
        "clusters":   df_pca.to_dict(orient='records'),
        "total":      len(df_pca),
        "n_clusters": int(df_pca['Cluster'].nunique()),
    }


@app.get("/dashboard/stats")
def get_stats():
    df_tr  = pd.read_csv('./data/processed/train.csv')
    df_vl  = pd.read_csv('./data/processed/val.csv')
    df_te  = pd.read_csv('./data/processed/test.csv')
    df_unl = pd.read_csv('./data/processed/unlabeled.csv')
    total  = len(df_tr) + len(df_vl) + len(df_te)

    return {
        "total_lignes":      total,
        "total_avec_unlab":  total + len(df_unl),
        "non_labellises":    len(df_unl),
        "nan_apres":         0,
        "train":             len(df_tr),
        "val":               len(df_vl),
        "test":              len(df_te),
        "categories":        list(df_tr['Categorie'].dropna().unique()),
        "distribution":      df_tr['Categorie'].value_counts().to_dict(),
    }


@app.get("/dashboard/models")
def get_models_performance():
    return {
        "meilleur_modele": "Random Forest",
        "accuracy_test":   0.9587,
        "f1_test":         0.9587,
        "comparaison": [
            {"nom": "Random Forest",       "accuracy": 95.87},
            {"nom": "KNN",                 "accuracy": 95.81},
            {"nom": "XGBoost",             "accuracy": 95.75},
            {"nom": "Decision Tree",       "accuracy": 95.68},
            {"nom": "Logistic Regression", "accuracy": 95.49},
            {"nom": "SVM",                 "accuracy": 95.37},
        ],
        "k_optimal":        4,
        "silhouette_score": 0.42,
        "variance_pca":     68.3,
        "ari_score":        0.31,
        "nlp_comparaison": [
            {"vectorisation": "TF-IDF",   "classifieur": "LinearSVC",    "f1": 0.89},
            {"vectorisation": "TF-IDF",   "classifieur": "RandomForest", "f1": 0.87},
            {"vectorisation": "Word2Vec", "classifieur": "LogReg",       "f1": 0.83},
            {"vectorisation": "FastText", "classifieur": "LogReg",       "f1": 0.82},
            {"vectorisation": "BoW",      "classifieur": "LinearSVC",    "f1": 0.79},
        ],
        "pretraitement": {
            "outliers_traites":    ["Volume", "Poids", "Opacite", "Prix_Revente"],
            "methode_outliers":    "IQR + Clipping",
            "features_originales": 6,
            "features_apres_fe":   6,
            "imputation_rmse": {
                "Mediane":          0.8234,
                "KNN (k=5)":        0.7891,
                "IterativeImputer": 0.7123,
            },
            "methode_choisie": "IterativeImputer",
        }
    }


# ── Endpoint diagnostic — pour débugger ──
@app.get("/debug/scaler")
def debug_scaler():
    """
    Endpoint de diagnostic pour vérifier le scaler.
    Ouvre http://localhost:8000/debug/scaler dans le navigateur.
    """
    df = pd.read_csv('./data/processed/train.csv')
    cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    return {
        "scaler_mean":   scaler.mean_.tolist(),
        "scaler_scale":  scaler.scale_.tolist(),
        "train_mean":    df[cols].mean().tolist(),
        "train_std":     df[cols].std().tolist(),
        "train_min":     df[cols].min().tolist(),
        "train_max":     df[cols].max().tolist(),
        "exemple_ligne": df[['Poids','Volume','Conductivite','Opacite','Rigidite','Source_enc']].iloc[0].tolist(),
        "categories":    list(le.classes_),
    }