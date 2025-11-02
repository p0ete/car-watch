"""Module pour gérer l'upload et le download des modèles vers/depuis Google Drive."""
import os
import pickle
import json
import logging
from io import BytesIO
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from config import GOOGLE_SHEETS_CREDENTIALS

# Configuration du logging
logger = logging.getLogger(__name__)

# ID du dossier Google Drive où stocker les modèles
DRIVE_FOLDER_ID = "1-9FfMbptqLGKTZNIOCvC3GrUQjwqM2tN"
USER_TO_IMPERSONATE = "paulineb.writing@gmail.com"

# Définir les scopes nécessaires
SCOPES = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive'
    ]

def authenticate():
    """
    Authentification simplifiée pour production.
    Nécessite un token.pickle pré-généré en local.
    """
    if not os.path.exists('token.pickle'):
        raise FileNotFoundError(
            "Le fichier 'token.pickle' est manquant. "
            "Générez-le en local avec generate_token.py"
        )
    
    # Charger les credentials
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
    
    # Rafraîchir automatiquement si expiré
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Sauvegarder le token rafraîchi
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def get_drive_service():
    """Crée un client Google Drive.
    
    Returns:
        Service Google Drive authentifié
    """
    logger.info("Connexion à Google Drive...")
    
    
    
    # S'authentifier
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    logger.info(f"✓ Connexion à Google Drive établie. Action au nom de : {USER_TO_IMPERSONATE}")
    return service


def upload_file_to_drive(local_filepath, drive_filename=None, folder_id=DRIVE_FOLDER_ID):
    """Upload un fichier vers Google Drive.
    
    Args:
        local_filepath: Chemin du fichier local à uploader
        drive_filename: Nom du fichier dans Google Drive (optionnel, utilise le nom local par défaut)
        folder_id: ID du dossier Google Drive de destination
        
    Returns:
        Dict avec les informations du fichier uploadé (id, name, webViewLink)
    """
    if drive_filename is None:
        drive_filename = local_filepath.split('/')[-1]
    
    logger.info(f"Upload de '{local_filepath}' vers Google Drive...")
    
    try:
        service = get_drive_service()
        
        # Vérifier si un fichier avec le même nom existe déjà dans le dossier
        query = f"name='{drive_filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        existing_files = results.get('files', [])
        
        if existing_files:
            # Supprimer l'ancien fichier
            file_id = existing_files[0]['id']
            logger.info(f"  - Fichier existant trouvé (ID: {file_id}), suppression...")
            service.files().delete(fileId=file_id).execute()
            logger.info(f"  - Ancien fichier supprimé")
        
        # Métadonnées du fichier
        file_metadata = {
            'name': drive_filename,
            'parents': [folder_id],
            'supportsAllDrives': True
        }
        
        # Upload du fichier
        media = MediaFileUpload(local_filepath, mimetype='application/octet-stream', resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        logger.info(f"✓ Fichier uploadé avec succès")
        logger.info(f"  - ID: {file.get('id')}")
        logger.info(f"  - Nom: {file.get('name')}")
        logger.info(f"  - Lien: {file.get('webViewLink')}")
        
        return {
            'id': file.get('id'),
            'name': file.get('name'),
            'webViewLink': file.get('webViewLink')
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'upload vers Google Drive: {e}", exc_info=True)
        raise


def download_file_from_drive(drive_filename, local_filepath, folder_id=DRIVE_FOLDER_ID):
    """Télécharge un fichier depuis Google Drive.
    
    Args:
        drive_filename: Nom du fichier dans Google Drive
        local_filepath: Chemin où sauvegarder le fichier localement
        folder_id: ID du dossier Google Drive source
        
    Returns:
        True si le téléchargement a réussi, False sinon
    """
    logger.info(f"Téléchargement de '{drive_filename}' depuis Google Drive...")
    
    try:
        service = get_drive_service()
        
        # Rechercher le fichier dans le dossier
        query = f"name='{drive_filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if not files:
            logger.error(f"Fichier '{drive_filename}' introuvable dans Google Drive")
            return False
        
        file_id = files[0]['id']
        logger.info(f"  - Fichier trouvé (ID: {file_id})")
        
        # Télécharger le fichier
        request = service.files().get_media(fileId=file_id)
        with open(local_filepath, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.info(f"  - Téléchargement: {int(status.progress() * 100)}%")
        
        logger.info(f"✓ Fichier téléchargé avec succès vers '{local_filepath}'")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement depuis Google Drive: {e}", exc_info=True)
        return False


def list_files_in_drive(folder_id=DRIVE_FOLDER_ID):
    """Liste tous les fichiers dans le dossier Google Drive.
    
    Args:
        folder_id: ID du dossier Google Drive
        
    Returns:
        Liste de dictionnaires avec les informations des fichiers
    """
    logger.info("Récupération de la liste des fichiers dans Google Drive...")
    
    try:
        service = get_drive_service()
        
        # Rechercher tous les fichiers dans le dossier
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, createdTime, modifiedTime, size)",
            orderBy="modifiedTime desc"
        ).execute()
        files = results.get('files', [])
        
        logger.info(f"✓ {len(files)} fichier(s) trouvé(s)")
        
        return files
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des fichiers: {e}", exc_info=True)
        return []


def upload_model_to_drive(source):
    """Upload les fichiers du modèle (.pkl et .json) vers Google Drive.
    
    Cette fonction upload à la fois le fichier .pkl et le fichier .json
    pour une source donnée.
    
    Args:
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        Dict avec les informations des fichiers uploadés
    """
    logger.info(f"Upload du modèle {source.upper()} vers Google Drive...")
    
    # Déterminer les noms de fichiers selon la source
    if source == 'leboncoin':
        pkl_filename = 'leboncoin_price_model.pkl'
        json_filename = 'leboncoin_price_model_metadata.json'
    elif source == 'annonces':
        pkl_filename = 'annonces_price_model.pkl'
        json_filename = 'annonces_price_model_metadata.json'
    else:
        raise ValueError(f"Source invalide: {source}")
    
    results = {}
    
    # Upload du fichier .pkl
    try:
        pkl_result = upload_file_to_drive(pkl_filename)
        results['model'] = pkl_result
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du fichier .pkl: {e}")
        raise
    
    # Upload du fichier .json
    try:
        json_result = upload_file_to_drive(json_filename)
        results['metadata'] = json_result
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du fichier .json: {e}")
        raise
    
    logger.info(f"✓ Modèle {source.upper()} uploadé avec succès vers Google Drive")
    
    return results


def download_model_from_drive(source):
    """Télécharge les fichiers du modèle (.pkl et .json) depuis Google Drive.
    
    Cette fonction télécharge à la fois le fichier .pkl et le fichier .json
    pour une source donnée.
    
    Args:
        source: 'leboncoin' ou 'annonces'
        
    Returns:
        True si le téléchargement a réussi, False sinon
    """
    logger.info(f"Téléchargement du modèle {source.upper()} depuis Google Drive...")
    
    # Déterminer les noms de fichiers selon la source
    if source == 'leboncoin':
        pkl_filename = 'leboncoin_price_model.pkl'
        json_filename = 'leboncoin_price_model_metadata.json'
    elif source == 'annonces':
        pkl_filename = 'annonces_price_model.pkl'
        json_filename = 'annonces_price_model_metadata.json'
    else:
        raise ValueError(f"Source invalide: {source}")
    
    # Télécharger le fichier .pkl
    pkl_success = download_file_from_drive(pkl_filename, pkl_filename)
    if not pkl_success:
        logger.error("Échec du téléchargement du fichier .pkl")
        return False
    
    # Télécharger le fichier .json
    json_success = download_file_from_drive(json_filename, json_filename)
    if not json_success:
        logger.error("Échec du téléchargement du fichier .json")
        return False
    
    logger.info(f"✓ Modèle {source.upper()} téléchargé avec succès depuis Google Drive")
    
    return True


def delete_old_models_in_drive(source, keep_last=5):
    """Supprime les anciens modèles dans Google Drive (garde les N derniers).
    
    Note: Cette fonction n'est pas encore implémentée car nous n'avons pas
    encore de système de versioning avec timestamps dans les noms de fichiers.
    
    Args:
        source: 'leboncoin' ou 'annonces'
        keep_last: Nombre de modèles à conserver
    """
    logger.warning("Fonction delete_old_models_in_drive() non implémentée (versioning non activé)")
    pass
