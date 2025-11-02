#!/usr/bin/env python3
"""
Module d'entraînement du modèle de prédiction de prix.
Support multi-sources : LeBonCoin et Annonces-Automobile
Récupération des données depuis Google Sheets
Peut être utilisé en ligne de commande ou appelé comme fonction depuis l'API
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import logging
import re
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from config import SPREADSHEET_ID, GOOGLE_SHEETS_CREDENTIALS
from google_drive import upload_model_to_drive

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION PAR SOURCE
# ============================================================================

SOURCE_CONFIG = {
    'leboncoin': {
        'worksheet_name': 'leboncoin_raw',
        'model_filename': 'leboncoin_price_model.pkl',
        'metadata_filename': 'leboncoin_price_model_metadata.json',
        'columns': {
            'price': 'price',
            'brand': 'brand',
            'model': 'model',
            'year': 'year',
            'mileage': 'mileage',
            'fuel': 'fuel',
            'gearbox': 'gearbox',
            'horsepower': 'horse_power_din'
        },
        'price_min': 500,
        'price_max': 200000
    },
    'annonces': {
        'worksheet_name': 'annonces-automobiles_raw',
        'model_filename': 'annonces_price_model.pkl',
        'metadata_filename': 'annonces_price_model_metadata.json',
        'columns': {
            'price': 'prix',
            'mileage': 'kilometrage',
            'year': 'annee',
            'gearbox': 'boite',
            'fuel': 'energie',
            'category': 'categorie'
        },
        'price_min': 500,
        'price_max': 200000
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


def get_sheets_client():
    """Crée un client Google Sheets.
    
    Returns:
        Client gspread authentifié
    """
    logger.info("Connexion à Google Sheets...")
    
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    try:
        # Essayer d'abord de charger depuis la variable d'environnement
        if GOOGLE_SHEETS_CREDENTIALS:
            logger.info("Chargement des credentials depuis la variable d'environnement")
            creds_data = json.loads(GOOGLE_SHEETS_CREDENTIALS)
        else:
            # Sinon, charger depuis le fichier local
            logger.info("Chargement des credentials depuis le fichier local")
            with open('google_sheets_credentials.json', 'r') as f:
                creds_data = json.load(f)
    except FileNotFoundError:
        logger.error("Fichier de credentials Google Sheets introuvable")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Erreur lors du parsing des credentials JSON: {e}")
        raise
    
    credentials = Credentials.from_service_account_info(creds_data, scopes=scopes)
    client = gspread.authorize(credentials)
    logger.info("✓ Connexion à Google Sheets établie")
    return client


# ============================================================================
# CHARGEMENT ET NETTOYAGE DES DONNÉES
# ============================================================================

def load_data_from_sheets(source):
    """Charge les données depuis Google Sheets.
    
    Args:
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        DataFrame pandas avec les données brutes
    """
    config = SOURCE_CONFIG[source]
    worksheet_name = config['worksheet_name']
    
    logger.info(f"Chargement des données depuis le worksheet '{worksheet_name}'...")
    
    # Connexion à Google Sheets
    client = get_sheets_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    
    # Récupération du worksheet
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        logger.error(f"Worksheet '{worksheet_name}' introuvable")
        raise
    
    # Récupération des données
    records = worksheet.get_all_records()
    df = pd.DataFrame(records)
    
    logger.info(f"✓ {len(df)} lignes chargées depuis Google Sheets")
    return df


def clean_data_leboncoin(df, config):
    """Nettoie les données LeBonCoin pour l'entraînement.
    
    Args:
        df: DataFrame brut depuis Google Sheets
        config: Configuration de la source
        
    Returns:
        DataFrame nettoyé
    """
    logger.info("Nettoyage des données LeBonCoin...")
    
    cols = config['columns']
    
    # Sélection des colonnes utiles
    required_cols = [cols['price'], cols['brand'], cols['model'], cols['year'], 
                     cols['mileage'], cols['fuel'], cols['gearbox'], cols['horsepower']]
    
    # Vérifier que toutes les colonnes existent
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes: {missing_cols}")
    
    available_cols = [col for col in required_cols if col in df.columns]
    df_clean = df[available_cols].copy()
    
    logger.info(f"  - Colonnes sélectionnées: {len(available_cols)}")
    
    # Renommage des colonnes pour uniformiser
    rename_map = {v: k for k, v in cols.items()}
    df_clean.rename(columns=rename_map, inplace=True)
    
    # Nettoyage des colonnes numériques
    logger.info("  - Nettoyage des colonnes numériques...")
    for col in ['price', 'year', 'mileage', 'horsepower']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(extract_numeric)
    
    # Suppression des lignes avec des valeurs manquantes
    initial_count = len(df_clean)
    df_clean.dropna(inplace=True)
    logger.info(f"  - Lignes après suppression des NaN: {len(df_clean)} (supprimé: {initial_count - len(df_clean)})")
    
    # Conversion en entiers
    for col in ['price', 'year', 'mileage', 'horsepower']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    
    # Filtrage des prix aberrants
    prix_min = config['price_min']
    prix_max = config['price_max']
    initial_count = len(df_clean)
    df_clean = df_clean[(df_clean['price'] >= prix_min) & (df_clean['price'] <= prix_max)]
    logger.info(f"  - Lignes après filtrage des prix ({prix_min}€ - {prix_max}€): {len(df_clean)} (supprimé: {initial_count - len(df_clean)})")
    
    logger.info(f"✓ Nettoyage terminé: {len(df_clean)} lignes prêtes pour l'entraînement")
    
    return df_clean


def clean_data_annonces(df, config):
    """Nettoie les données Annonces-Automobile pour l'entraînement.
    
    Args:
        df: DataFrame brut depuis Google Sheets
        config: Configuration de la source
        
    Returns:
        DataFrame nettoyé
    """
    logger.info("Nettoyage des données Annonces-Automobile...")
    
    cols = config['columns']
    
    # Sélection des colonnes utiles
    required_cols = [cols['price'], cols['mileage'], cols['year'], 
                     cols['gearbox'], cols['fuel'], cols['category']]
    
    # Vérifier que toutes les colonnes existent
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes: {missing_cols}")
    
    available_cols = [col for col in required_cols if col in df.columns]
    df_clean = df[available_cols].copy()
    
    logger.info(f"  - Colonnes sélectionnées: {len(available_cols)}")
    
    # Renommage des colonnes pour uniformiser
    rename_map = {v: k for k, v in cols.items()}
    df_clean.rename(columns=rename_map, inplace=True)
    
    # Nettoyage des colonnes numériques
    logger.info("  - Nettoyage des colonnes numériques...")
    df_clean['price'] = df_clean['price'].apply(extract_numeric)
    df_clean['mileage'] = df_clean['mileage'].apply(extract_numeric)
    df_clean['year'] = df_clean['year'].apply(extract_year)
    
    # Suppression des lignes avec des valeurs manquantes
    initial_count = len(df_clean)
    df_clean.dropna(subset=['price', 'mileage', 'year'], inplace=True)
    logger.info(f"  - Lignes après suppression des NaN: {len(df_clean)} (supprimé: {initial_count - len(df_clean)})")
    
    # Conversion en entiers
    df_clean['price'] = df_clean['price'].astype(int)
    df_clean['mileage'] = df_clean['mileage'].astype(int)
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filtrage des prix aberrants
    prix_min = config['price_min']
    prix_max = config['price_max']
    initial_count = len(df_clean)
    df_clean = df_clean[(df_clean['price'] >= prix_min) & (df_clean['price'] <= prix_max)]
    logger.info(f"  - Lignes après filtrage des prix ({prix_min}€ - {prix_max}€): {len(df_clean)} (supprimé: {initial_count - len(df_clean)})")
    
    logger.info(f"✓ Nettoyage terminé: {len(df_clean)} lignes prêtes pour l'entraînement")
    
    return df_clean


# ============================================================================
# PRÉPARATION DES FEATURES
# ============================================================================

def prepare_features_leboncoin(df):
    """Prépare les features pour le modèle LeBonCoin.
    
    Args:
        df: DataFrame nettoyé
        
    Returns:
        Tuple (X, y, feature_info)
    """
    logger.info("Préparation des features LeBonCoin...")
    
    # Séparation de la cible
    y = df['price']
    
    # Création des features
    X = pd.get_dummies(
        df.drop('price', axis=1),
        columns=['brand', 'model', 'fuel', 'gearbox'],
        drop_first=True
    )
    
    logger.info(f"  - Features créées: {X.shape[1]} colonnes")
    logger.info(f"  - Exemples de features: {list(X.columns[:5])}")
    
    feature_info = {
        'feature_columns': list(X.columns),
        'num_features': X.shape[1]
    }
    
    return X, y, feature_info


def prepare_features_annonces(df):
    """Prépare les features pour le modèle Annonces-Automobile.
    
    Args:
        df: DataFrame nettoyé
        
    Returns:
        Tuple (X, y, feature_info)
    """
    logger.info("Préparation des features Annonces-Automobile...")
    
    # Séparation de la cible
    y = df['price']
    
    # Calcul de médianes pour l'imputation (sauvegardées pour la prédiction)
    km_median = df['mileage'].median()
    annee_median = df['year'].median()
    
    logger.info(f"  - Médiane kilométrage: {km_median:.0f} km")
    logger.info(f"  - Médiane année: {annee_median:.0f}")
    
    # Création des features
    features = pd.DataFrame()
    
    # Features numériques
    features['kilometrage'] = df['mileage']
    features['annee'] = df['year']
    features['age'] = 2025 - df['year']
    
    # Encodage de la boîte de vitesses
    boite_mapping = {'Automatique': 1, 'Mecanique': 0, 'BVA': 1}
    features['boite_auto'] = df['gearbox'].map(boite_mapping).fillna(0)
    
    # Encodage one-hot de l'énergie
    energie_dummies = pd.get_dummies(df['fuel'], prefix='energie', drop_first=False)
    for col in energie_dummies.columns:
        features[col] = energie_dummies[col]
    
    # Encodage one-hot de la catégorie
    categorie_dummies = pd.get_dummies(df['category'], prefix='categorie', drop_first=False)
    for col in categorie_dummies.columns:
        features[col] = categorie_dummies[col]
    
    X = features
    
    logger.info(f"  - Features créées: {X.shape[1]} colonnes")
    logger.info(f"  - Exemples de features: {list(X.columns[:5])}")
    
    feature_info = {
        'feature_columns': list(X.columns),
        'num_features': X.shape[1],
        'km_median': km_median,
        'annee_median': annee_median
    }
    
    return X, y, feature_info


# ============================================================================
# ENTRAÎNEMENT DU MODÈLE
# ============================================================================

def train_model(X, y, source):
    """Entraîne le modèle de prédiction.
    
    Args:
        X: Features
        y: Target (prix)
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        Tuple (model, metrics)
    """
    logger.info(f"Entraînement du modèle {source.upper()}...")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"  - Ensemble d'entraînement: {len(X_train)} annonces")
    logger.info(f"  - Ensemble de test: {len(X_test)} annonces")
    
    # Entraînement du modèle
    logger.info("  - Entraînement en cours...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("  ✓ Entraînement terminé")
    
    # Évaluation sur l'ensemble de test
    logger.info("  - Évaluation du modèle...")
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("MÉTRIQUES D'ÉVALUATION DU MODÈLE")
    logger.info("=" * 60)
    logger.info(f"R² (coefficient de détermination): {r2:.4f}")
    logger.info(f"RMSE (erreur quadratique moyenne): {rmse:.2f} €")
    logger.info(f"MAE (erreur absolue moyenne): {mae:.2f} €")
    logger.info("=" * 60)
    logger.info("")
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }
    
    return model, metrics


# ============================================================================
# SAUVEGARDE DU MODÈLE ET DES MÉTADONNÉES
# ============================================================================

def save_model_and_metadata(model, feature_info, metrics, source, df_clean):
    """Sauvegarde le modèle et les métadonnées.
    
    Args:
        model: Modèle entraîné
        feature_info: Informations sur les features
        metrics: Métriques d'évaluation
        source: 'leboncoin' ou 'annonces'
        df_clean: DataFrame nettoyé utilisé pour l'entraînement
    """
    config = SOURCE_CONFIG[source]
    model_filename = config['model_filename']
    metadata_filename = config['metadata_filename']
    
    logger.info("Sauvegarde du modèle et des métadonnées...")
    
    # Préparation des données du modèle
    model_data = {
        'model': model,
        'feature_columns': feature_info['feature_columns']
    }
    
    # Ajouter les médianes pour Annonces-Automobile
    if source == 'annonces':
        model_data['km_median'] = feature_info['km_median']
        model_data['annee_median'] = feature_info['annee_median']
        model_data['metrics'] = metrics
    
    # Sauvegarde du modèle
    logger.info(f"  - Sauvegarde du modèle dans '{model_filename}'...")
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"  ✓ Modèle sauvegardé")
    
    # Préparation des métadonnées
    metadata = {
        'source': source,
        'trained_at': datetime.now().isoformat(),
        'training_samples': len(df_clean),
        'num_features': feature_info['num_features'],
        'feature_columns': feature_info['feature_columns'],
        'metrics': {
            'R2': float(metrics['R2']),
            'RMSE': float(metrics['RMSE']),
            'MAE': float(metrics['MAE'])
        },
        'price_range': {
            'min': int(df_clean['price'].min()),
            'max': int(df_clean['price'].max()),
            'mean': float(df_clean['price'].mean()),
            'median': float(df_clean['price'].median())
        },
        'config': {
            'worksheet_name': config['worksheet_name'],
            'price_min_filter': config['price_min'],
            'price_max_filter': config['price_max']
        }
    }
    
    # Ajouter les médianes pour Annonces-Automobile
    if source == 'annonces':
        metadata['imputation'] = {
            'km_median': float(feature_info['km_median']),
            'annee_median': float(feature_info['annee_median'])
        }
    
    # Sauvegarde des métadonnées
    logger.info(f"  - Sauvegarde des métadonnées dans '{metadata_filename}'...")
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Métadonnées sauvegardées")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("FICHIERS GÉNÉRÉS")
    logger.info("=" * 60)
    logger.info(f"Modèle: {model_filename}")
    logger.info(f"Métadonnées: {metadata_filename}")
    logger.info("=" * 60)


# ============================================================================
# FONCTION PRINCIPALE POUR L'API
# ============================================================================

def train_model_for_source(source):
    """Entraîne un modèle pour une source donnée.
    
    Cette fonction peut être appelée depuis l'API ou en ligne de commande.
    
    Args:
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        Dict contenant les résultats de l'entraînement
        
    Raises:
        ValueError: Si la source n'est pas valide
        Exception: En cas d'erreur lors de l'entraînement
    """
    # Validation de la source
    if source not in ['leboncoin', 'annonces']:
        raise ValueError(f"Source invalide: {source}. Doit être 'leboncoin' ou 'annonces'")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"ENTRAÎNEMENT DU MODÈLE - SOURCE: {source.upper()}")
    logger.info("=" * 60)
    logger.info("")
    
    try:
        # 1. Chargement des données depuis Google Sheets
        df_raw = load_data_from_sheets(source)
        
        # 2. Nettoyage des données
        config = SOURCE_CONFIG[source]
        if source == 'leboncoin':
            df_clean = clean_data_leboncoin(df_raw, config)
        else:  # annonces
            df_clean = clean_data_annonces(df_raw, config)
        
        if df_clean.empty:
            raise Exception("Aucune donnée après nettoyage. Entraînement annulé.")
        
        # 3. Préparation des features
        if source == 'leboncoin':
            X, y, feature_info = prepare_features_leboncoin(df_clean)
        else:  # annonces
            X, y, feature_info = prepare_features_annonces(df_clean)
        
        # 4. Entraînement du modèle
        model, metrics = train_model(X, y, source)
        
        # 5. Sauvegarde du modèle et des métadonnées (localement)
        save_model_and_metadata(model, feature_info, metrics, source, df_clean)
        
        # 6. Upload vers Google Drive
        logger.info("")
        logger.info("=" * 60)
        logger.info("UPLOAD VERS GOOGLE DRIVE")
        logger.info("=" * 60)
        logger.info("")
        
        try:
            drive_results = upload_model_to_drive(source)
            logger.info("")
            logger.info("✓ Fichiers uploadés vers Google Drive:")
            logger.info(f"  - Modèle (.pkl): {drive_results['model']['webViewLink']}")
            logger.info(f"  - Métadonnées (.json): {drive_results['metadata']['webViewLink']}")
        except Exception as e:
            logger.error(f"Erreur lors de l'upload vers Google Drive: {e}", exc_info=True)
            logger.warning("⚠️ Les fichiers sont sauvegardés localement mais pas sur Google Drive")
            drive_results = None
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info("")
        
        # Retourner les résultats pour l'API
        result = {
            'success': True,
            'source': source,
            'training_samples': len(df_clean),
            'num_features': feature_info['num_features'],
            'metrics': {
                'R2': float(metrics['R2']),
                'RMSE': float(metrics['RMSE']),
                'MAE': float(metrics['MAE'])
            },
            'model_filename': config['model_filename'],
            'metadata_filename': config['metadata_filename']
        }
        
        # Ajouter les informations Google Drive si l'upload a réussi
        if drive_results:
            result['google_drive'] = {
                'model_id': drive_results['model']['id'],
                'model_link': drive_results['model']['webViewLink'],
                'metadata_id': drive_results['metadata']['id'],
                'metadata_link': drive_results['metadata']['webViewLink']
            }
        
        return result
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("✗ ERREUR LORS DE L'ENTRAÎNEMENT")
        logger.error("=" * 60)
        logger.error(f"Erreur: {e}", exc_info=True)
        logger.error("")
        raise


# ============================================================================
# POINT D'ENTRÉE EN LIGNE DE COMMANDE
# ============================================================================

def main():
    """Point d'entrée principal du script en ligne de commande."""
    
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description='Entraînement du modèle de prédiction de prix de véhicules'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['leboncoin', 'annonces'],
        help='Source des données (leboncoin ou annonces)'
    )
    
    args = parser.parse_args()
    
    # Appel de la fonction principale
    try:
        result = train_model_for_source(args.source)
        print("\n✓ Entraînement réussi!")
        print(f"  - Modèle: {result['model_filename']}")
        print(f"  - Métadonnées: {result['metadata_filename']}")
    except Exception as e:
        print(f"\n✗ Échec de l'entraînement: {e}")
        exit(1)


if __name__ == "__main__":
    main()