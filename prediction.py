"""Module de prédiction des prix de véhicules.
Support multi-sources : LeBonCoin et Annonces-Automobile
Téléchargement automatique des modèles depuis Google Drive
"""

import os
import pandas as pd
import numpy as np
import pickle
import re
import logging
from datetime import datetime
from google_drive import download_model_from_drive

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION PAR SOURCE
# ============================================================================

SOURCE_CONFIG = {
    'leboncoin': {
        'model_filename': 'leboncoin_price_model.pkl',
        'metadata_filename': 'leboncoin_price_model_metadata.json'
    },
    'annonces': {
        'model_filename': 'annonces_price_model.pkl',
        'metadata_filename': 'annonces_price_model_metadata.json'
    }
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def extract_numeric(value):
    """Extrait la valeur numérique d'une chaîne de caractères.
    
    Args:
        value: Valeur à convertir (peut être str, int, float, ou NaN)
        
    Returns:
        float ou np.nan
    """
    if pd.isna(value):
        return np.nan
    
    # Si c'est déjà un nombre, le retourner
    if isinstance(value, (int, float)):
        return float(value)
    
    # Sinon, extraire le nombre de la chaîne
    value_str = str(value).replace(' ', '').replace(',', '.')
    match = re.search(r'(\d+\.?\d*)', value_str)
    return float(match.group(1)) if match else np.nan


def extract_year(value):
    """Extrait l'année d'une chaîne de caractères.
    
    Args:
        value: Valeur contenant l'année
        
    Returns:
        int ou np.nan
    """
    if pd.isna(value):
        return np.nan
    
    # Si c'est déjà un nombre dans la plage valide
    if isinstance(value, (int, float)):
        year = int(value)
        return year if 1990 <= year <= 2025 else np.nan
    
    # Sinon, extraire l'année de la chaîne
    value_str = str(value)
    match = re.search(r'(\d{4})', value_str)
    if match:
        year = int(match.group(1))
        return year if 1990 <= year <= 2025 else np.nan
    return np.nan


def ensure_model_downloaded(source):
    """S'assure que le modèle est téléchargé depuis Google Drive.
    
    Args:
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        True si le modèle est disponible, False sinon
    """
    config = SOURCE_CONFIG[source]
    model_filename = config['model_filename']
    
    # Vérifier si le modèle existe déjà localement
    if os.path.exists(model_filename):
        logger.info(f"Modèle {source} déjà présent localement")
        return True
    
    # Sinon, télécharger depuis Google Drive
    logger.info(f"Modèle {source} non trouvé localement, téléchargement depuis Google Drive...")
    try:
        success = download_model_from_drive(source)
        if success:
            logger.info(f"✓ Modèle {source} téléchargé avec succès")
            return True
        else:
            logger.error(f"✗ Échec du téléchargement du modèle {source}")
            return False
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle {source}: {e}", exc_info=True)
        return False


# ============================================================================
# PRÉDICTION LEBONCOIN
# ============================================================================

def prepare_features_leboncoin(df, feature_columns):
    """Prépare les features pour la prédiction LeBonCoin.
    
    Args:
        df: DataFrame avec les données brutes
        feature_columns: Liste des colonnes de features attendues par le modèle
        
    Returns:
        DataFrame avec les features préparées
    """
    df_work = df.copy()
    
    # Conversion des colonnes numériques
    for col in ['year', 'mileage', 'horse_power_din']:
        if col in df_work.columns:
            df_work[col] = df_work[col].apply(extract_numeric)
    
    # One-hot encoding
    X = pd.get_dummies(
        df_work[['brand', 'model', 'year', 'mileage', 'fuel', 'gearbox', 'horse_power_din']],
        columns=['brand', 'model', 'fuel', 'gearbox'],
        drop_first=True
    )
    
    # S'assurer que toutes les colonnes du training sont présentes
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Garder uniquement les colonnes du training dans le bon ordre
    X = X[feature_columns]
    
    return X


def add_predicted_price_leboncoin(df):
    """Ajoute la colonne predicted_price pour LeBonCoin.
    
    Args:
        df: DataFrame avec les données LeBonCoin
        
    Returns:
        DataFrame avec la colonne predicted_price ajoutée
    """
    logger.info("Prédiction des prix pour LeBonCoin...")
    
    # S'assurer que le modèle est disponible
    if not ensure_model_downloaded('leboncoin'):
        logger.error("Impossible de charger le modèle LeBonCoin")
        df['predicted_price'] = np.nan
        return df
    
    try:
        # Charger le modèle
        with open('leboncoin_price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        logger.info(f"Modèle LeBonCoin chargé ({len(feature_columns)} features)")
        
        # Préparer les features
        X = prepare_features_leboncoin(df, feature_columns)
        
        # Prédiction
        predictions = model.predict(X)
        
        # Ajouter la colonne predicted_price
        df['predicted_price'] = np.maximum(0, predictions).round()
        
        logger.info(f"✓ Prédiction terminée pour {len(df)} annonces LeBonCoin")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction LeBonCoin: {e}", exc_info=True)
        df['predicted_price'] = np.nan
        return df


# ============================================================================
# PRÉDICTION ANNONCES-AUTOMOBILE
# ============================================================================

def prepare_features_annonces(df, feature_columns, km_median, annee_median):
    """Prépare les features pour la prédiction Annonces-Automobile.
    
    Args:
        df: DataFrame avec les données brutes
        feature_columns: Liste des colonnes de features attendues par le modèle
        km_median: Médiane du kilométrage pour l'imputation
        annee_median: Médiane de l'année pour l'imputation
        
    Returns:
        DataFrame avec les features préparées
    """
    df_work = df.copy()
    
    # Nettoyage des colonnes numériques
    df_work['kilometrage_clean'] = df_work['kilometrage'].apply(extract_numeric)
    df_work['annee_clean'] = df_work['annee'].apply(extract_year)
    
    # Création des features
    features = pd.DataFrame()
    
    # Kilométrage (imputation par la médiane du training)
    features['kilometrage'] = df_work['kilometrage_clean'].fillna(km_median)
    
    # Année (imputation par la médiane du training)
    features['annee'] = df_work['annee_clean'].fillna(annee_median)
    
    # Âge du véhicule
    current_year = datetime.now().year
    features['age'] = current_year - features['annee']
    
    # Encodage de la boîte de vitesses
    boite_mapping = {'Automatique': 1, 'Mecanique': 0, 'BVA': 1}
    features['boite_auto'] = df_work['boite'].map(boite_mapping).fillna(0)
    
    # Encodage one-hot de l'énergie
    energie_dummies = pd.get_dummies(df_work['energie'], prefix='energie', drop_first=False)
    for col in energie_dummies.columns:
        features[col] = energie_dummies[col]
    
    # Encodage one-hot de la catégorie
    categorie_dummies = pd.get_dummies(df_work['categorie'], prefix='categorie', drop_first=False)
    for col in categorie_dummies.columns:
        features[col] = categorie_dummies[col]
    
    # S'assurer que toutes les colonnes du training sont présentes
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    
    # Garder uniquement les colonnes du training dans le bon ordre
    features = features[feature_columns]
    
    return features


def add_predicted_price_annonces(df):
    """Ajoute la colonne predicted_price pour Annonces-Automobile.
    
    Args:
        df: DataFrame avec les données Annonces-Automobile
        
    Returns:
        DataFrame avec la colonne predicted_price ajoutée
    """
    logger.info("Prédiction des prix pour Annonces-Automobile...")
    
    # S'assurer que le modèle est disponible
    if not ensure_model_downloaded('annonces'):
        logger.error("Impossible de charger le modèle Annonces-Automobile")
        df['predicted_price'] = np.nan
        return df
    
    try:
        # Charger le modèle
        with open('annonces_price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        km_median = model_data['km_median']
        annee_median = model_data['annee_median']
        
        logger.info(f"Modèle Annonces-Automobile chargé ({len(feature_columns)} features)")
        
        # Préparer les features
        X = prepare_features_annonces(df, feature_columns, km_median, annee_median)
        
        # Prédiction
        predictions = model.predict(X)
        
        # Ajouter la colonne predicted_price
        df['predicted_price'] = np.maximum(0, predictions).round()
        
        logger.info(f"✓ Prédiction terminée pour {len(df)} annonces Annonces-Automobile")
        
        return df
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction Annonces-Automobile: {e}", exc_info=True)
        df['predicted_price'] = np.nan
        return df


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def add_predicted_price_column(df, source):
    """Ajoute une colonne 'predicted_price' au DataFrame.
    
    Télécharge le modèle depuis Google Drive si nécessaire,
    puis prédit les prix selon la source.
    
    Args:
        df: DataFrame pandas contenant les données des véhicules
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        DataFrame avec la colonne 'predicted_price' ajoutée
    """
    if source not in ['leboncoin', 'annonces']:
        logger.error(f"Source invalide: {source}. Doit être 'leboncoin' ou 'annonces'")
        df['predicted_price'] = np.nan
        return df
    
    logger.info(f"Ajout de la colonne predicted_price pour {source.upper()}...")
    
    if source == 'leboncoin':
        return add_predicted_price_leboncoin(df)
    else:  # annonces
        return add_predicted_price_annonces(df)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Exemple pour LeBonCoin
    print("Exemple d'utilisation pour LeBonCoin:")
    df_leboncoin = pd.DataFrame({
        'brand': ['Peugeot', 'Renault'],
        'model': ['308', 'Clio'],
        'year': [2020, 2019],
        'mileage': [50000, 60000],
        'fuel': ['Diesel', 'Essence'],
        'gearbox': ['Manuelle', 'Automatique'],
        'horse_power_din': [130, 90]
    })
    
    df_leboncoin = add_predicted_price_column(df_leboncoin, 'leboncoin')
    print(df_leboncoin[['brand', 'model', 'year', 'predicted_price']])
    
    print("\n" + "="*60 + "\n")
    
    # Exemple pour Annonces-Automobile
    print("Exemple d'utilisation pour Annonces-Automobile:")
    df_annonces = pd.DataFrame({
        'kilometrage': ['50.000 km', '60.000 km'],
        'annee': ['2020', '2019'],
        'boite': ['Mecanique', 'Automatique'],
        'energie': ['Diesel', 'Essence'],
        'categorie': ['Berline', 'Citadine']
    })
    
    df_annonces = add_predicted_price_column(df_annonces, 'annonces')
    print(df_annonces[['kilometrage', 'annee', 'predicted_price']])