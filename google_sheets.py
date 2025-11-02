"""Service simple pour Google Sheets - approche originale simplifiée."""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import json
from datetime import datetime
import pytz
from config import SPREADSHEET_ID, GOOGLE_SHEETS_CREDENTIALS
from prediction import add_predicted_price_column
from config import ENABLE_PREDICTION
import logging 

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sheets_client():
    """Crée un client Google Sheets.
    
    Charge les credentials depuis la variable d'environnement ou depuis un fichier local.
    
    Returns:
        Client gspread authentifié
    """
    # Configuration des scopes nécessaires
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
    
    # Création du client avec les credentials
    credentials = Credentials.from_service_account_info(creds_data, scopes=scopes)
    return gspread.authorize(credentials)


def prepare_data_for_saving(data, source=None):
    """Prépare les données en ajoutant timestamp et prédictions.
    
    Args:
        data: DataFrame pandas contenant les données à préparer
        source: Source des données ('leboncoin' ou 'annonces'), optionnel
        
    Returns:
        DataFrame avec les colonnes supplémentaires ou None en cas d'erreur
    """
    if data.empty:
        return data
    
    # Ajout de la colonne scraped_at avec timestamp actuel
    logger.info("Ajout de la colonne scraped_at...")
    data = data.copy()
    data['scraped_at'] = datetime.now(pytz.timezone('Europe/Paris')).strftime('%d/%m/%Y %H:%M:%S')
    
    # Ajout des prédictions de prix (si activé et source fournie)
    if ENABLE_PREDICTION and source:
        logger.info(f"Ajout de la colonne predicted_price pour {source.upper()}...")
        try:
            df_with_predictions = add_predicted_price_column(data, source)
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
            df_with_predictions = data
    else:
        if not ENABLE_PREDICTION:
            logger.info("Prédiction désactivée (ENABLE_PREDICTION=False)")
        df_with_predictions = data
    
    return df_with_predictions


def get_or_create_worksheet(spreadsheet, worksheet_name):
    """Récupère ou crée un worksheet.
    
    Args:
        spreadsheet: Objet Spreadsheet de gspread
        worksheet_name: Nom du worksheet à récupérer ou créer
        
    Returns:
        Tuple (worksheet, existing_data)
    """
    try:
        # Essayer de récupérer le worksheet existant
        worksheet = spreadsheet.worksheet(worksheet_name)
        existing_data = get_existing_data(worksheet)
        logger.info(f"Worksheet '{worksheet_name}' trouvé avec {len(existing_data)} lignes existantes")
    except gspread.exceptions.WorksheetNotFound:
        # Créer un nouveau worksheet s'il n'existe pas
        logger.info(f"Création du nouveau worksheet '{worksheet_name}'")
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
        existing_data = pd.DataFrame()
    
    return worksheet, existing_data


def save_to_sheet(data, worksheet_name, id_column='id', source=None):
    """Sauvegarde les données dans Google Sheets.
    
    Args:
        data: DataFrame pandas contenant les données à sauvegarder
        worksheet_name: Nom du worksheet où sauvegarder les données
        id_column: Nom de la colonne utilisée comme identifiant unique (défaut: 'id')
        source: Source des données ('leboncoin' ou 'annonces'), optionnel
    """
    if data.empty:
        logger.info("La dataframe est vide. Rien à enregistrer.")
        return
    
    # Préparation des données (ajout timestamp et prédictions)
    df_with_predictions = prepare_data_for_saving(data, source=source)
    if df_with_predictions is None:
        logger.error("Erreur lors de la préparation des données")
        return

    # Connexion Google Sheets
    logger.info("Création du client Google Sheet...")
    try:
        client = get_sheets_client()
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Google Sheets: {e}", exc_info=True)
        return
    
    # Récupération/création du worksheet
    logger.info(f"Ouverture du worksheet '{worksheet_name}'...")
    worksheet, existing_data = get_or_create_worksheet(spreadsheet, worksheet_name)
    
    # Filtrage des nouvelles données pour éviter les doublons
    logger.info("Filtrage des nouvelles données...")
    new_data = filter_new_data(df_with_predictions, existing_data, id_column)
    logger.info(f"Il y a {len(new_data)} nouvelles annonces et {len(df_with_predictions) - len(new_data)} annonces déjà présentes dans le Google Sheet.")
    
    # Ajout des nouvelles données si nécessaire
    if not new_data.empty:
        append_new_data(worksheet, new_data, existing_data.empty)
    else:
        logger.info("Aucune nouvelle donnée à ajouter")


def get_existing_data(worksheet):
    """Récupère les données existantes du worksheet.
    
    Args:
        worksheet: Objet Worksheet de gspread
        
    Returns:
        DataFrame pandas contenant les données existantes
    """
    try:
        records = worksheet.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données existantes: {e}")
        return pd.DataFrame()


def filter_new_data(new_data, existing_data, id_column):
    """Filtre les nouvelles données pour éviter les doublons.
    
    Args:
        new_data: DataFrame des nouvelles données
        existing_data: DataFrame des données existantes
        id_column: Nom de la colonne utilisée comme identifiant unique
        
    Returns:
        DataFrame contenant uniquement les nouvelles données (sans doublons)
    """
    if existing_data.empty:
        return new_data
    
    if id_column in existing_data.columns:
        # Créer un set des IDs existants pour une recherche rapide
        existing_ids = set(existing_data[id_column].astype(str))
        # Filtrer les nouvelles données pour ne garder que celles qui ne sont pas déjà présentes
        mask = ~new_data[id_column].astype(str).isin(existing_ids)
        return new_data[mask]
    
    return new_data


def append_new_data(worksheet, new_data, is_empty_sheet):
    """Ajoute les nouvelles données au worksheet - approche originale.
    
    Args:
        worksheet: Objet Worksheet de gspread
        new_data: DataFrame des nouvelles données à ajouter
        is_empty_sheet: Booléen indiquant si le worksheet est vide (besoin d'ajouter les headers)
    """
    # Conversion des données en format string pour Google Sheets
    formatted_data = []
    logger.info("Formatage des données...")
    for _, row in new_data.iterrows():
        formatted_row = []
        for value in row:
            if pd.isna(value) or value is None:
                formatted_row.append("")
            else:
                formatted_row.append(str(value))
        formatted_data.append(formatted_row)
    
    if is_empty_sheet:
        # Première fois : ajouter les headers en première ligne
        headers = list(new_data.columns)
        rows_to_add = [headers] + formatted_data
        logger.info(f"Ajout des headers et de {len(formatted_data)} lignes de données")
    else:
        # Append seulement les données (headers déjà présents)
        rows_to_add = formatted_data
        logger.info(f"Ajout de {len(formatted_data)} lignes de données")
    
    # Envoi avec USER_ENTERED pour permettre la conversion automatique des types
    logger.info("Écriture dans le Google Sheet...")
    try:
        worksheet.append_rows(rows_to_add, value_input_option="USER_ENTERED")
        logger.info("Écriture terminée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture dans Google Sheets: {e}", exc_info=True)


def get_good_deals(worksheet_name, discount_threshold=0.8, km_max=None, start_date=None, end_date=None):
    """Récupère les véhicules avec un prix inférieur au seuil du prix prédit.
    
    Args:
        worksheet_name: Nom du worksheet à analyser
        discount_threshold: Seuil de remise (0.8 = prix < 80% du prix prédit)
        km_max: Kilométrage maximum (optionnel)
        start_date: Date de début au format DD/MM/YYYY (optionnel)
        end_date: Date de fin au format DD/MM/YYYY (optionnel)
        
    Returns:
        Liste de dictionnaires contenant les bonnes affaires
    """
    logger.info(f"Recherche des bonnes affaires dans '{worksheet_name}'...")
    
    try:
        client = get_sheets_client()
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Google Sheets: {e}", exc_info=True)
        return []
    
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
        data = get_existing_data(worksheet)
    except gspread.exceptions.WorksheetNotFound:
        logger.error(f"Worksheet '{worksheet_name}' non trouvé.")
        return []
    
    if data.empty:
        logger.info("Aucune donnée trouvée.")
        return []
    
    # Filtrage des bonnes affaires
    try:
        # Conversion en numérique pour les calculs
        data['price'] = pd.to_numeric(data['price'], errors='coerce')
        data['predicted_price'] = pd.to_numeric(data['predicted_price'], errors='coerce')
        
        # Filtrage par prix : prix réel < seuil * prix prédit
        good_deals = data[
            (data['price'].notna()) & 
            (data['predicted_price'].notna()) & 
            (data['price'] < discount_threshold * data['predicted_price'])
        ]
        
        # Filtrage par kilométrage si spécifié
        if km_max:
            data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')
            good_deals = good_deals[good_deals['mileage'] < km_max]
        
        # Filtrage par date si spécifié
        if start_date or end_date:
            good_deals['scraped_at'] = pd.to_datetime(
                good_deals['scraped_at'], 
                format='%d/%m/%Y %H:%M:%S', 
                errors='coerce'
            )
            
            if start_date:
                start_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
                good_deals = good_deals[good_deals['scraped_at'] >= start_dt]
            
            if end_date:
                # Inclure toute la journée de fin
                end_dt = pd.to_datetime(end_date, format='%d/%m/%Y') + pd.Timedelta(days=1)
                good_deals = good_deals[good_deals['scraped_at'] < end_dt]
        
        # Calcul de la différence de prix et tri
        good_deals['price_difference'] = good_deals['predicted_price'] - good_deals['price']
        good_deals.sort_values('price_difference', ascending=False, inplace=True)

        logger.info(f"Trouvé {len(good_deals)} bonnes affaires sur {len(data)} véhicules.")
        
        # Conversion en liste de dictionnaires pour l'API
        return good_deals.to_dict('records')
        
    except Exception as e:
        logger.error(f"Erreur lors du filtrage des bonnes affaires: {e}", exc_info=True)
        return []