"""Client simple pour récupérer les annonces LeBonCoin."""

import requests
import pandas as pd
from config import LEBONCOIN_API_URL, LEBONCOIN_HEADERS
import logging 
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_leboncoin_ads(limit):
    """Récupère les annonces LeBonCoin avec pagination automatique.
    
    Args:
        limit: Nombre maximum d'annonces à récupérer
        
    Returns:
        DataFrame pandas contenant les annonces extraites
    """
    logger.info(f"Récupération des données de Leboncoin (limit={limit})...")
    
    all_ads = []
    offset = 0
    page_size = 100  # Taille max supportée par l'API
    
    page_number = 0
    while offset < limit:
        page_number += 1
        # Calcul de la taille pour cette requête
        current_limit = min(page_size, limit - offset)
        
        try:
            # Requête API pour ce chunk
            logger.info(f"- Appel à l'API numéro {page_number} (limit={current_limit}, offset={offset})")
            payload = build_api_payload(current_limit, offset)
            response = requests.post(LEBONCOIN_API_URL, headers=LEBONCOIN_HEADERS, json=payload)
            response.raise_for_status()
            
            data = response.json()
            ads = data.get("ads", [])
            logger.info(f"- {len(ads)} annonces récupérées")

            if not ads:
                logger.info("- Aucune annonce supplémentaire disponible, arrêt de la pagination")
                break  # Plus d'annonces disponibles
            
            all_ads.extend(ads)
            
            # Si l'API retourne moins que demandé, c'est la fin
            if len(ads) < current_limit:
                logger.info("- Dernière page atteinte (moins d'annonces que demandé)")
                break
            
            offset += len(ads)
            
            # Pause pour éviter de surcharger l'API
            time.sleep(5)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de l'appel API (page {page_number}): {e}", exc_info=True)
            break
        except Exception as e:
            logger.error(f"Erreur inattendue lors du traitement (page {page_number}): {e}", exc_info=True)
            break

    logger.info(f"Au total, {len(all_ads)} annonces récupérées.")
    return extract_ad_data(all_ads)


def build_api_payload(limit, offset=0):
    """Construit le payload pour l'API LeBonCoin avec pagination.
    
    Args:
        limit: Nombre d'annonces à récupérer dans cette requête
        offset: Position de départ dans la liste totale des annonces
        
    Returns:
        Dict contenant le payload JSON pour l'API
    """
    return {
        "filters": {
            "category": {"id": "2"},  # Catégorie "Voitures"
            "enums": {"ad_type": ["offer"]},  # Type "Offre" (pas de demande)
            "keywords": {"text": ""}  # Pas de filtre par mot-clé
        },
        "sort_by": "time",  # Tri par date de publication
        "sort_order": "desc",  # Plus récentes en premier
        "limit": limit,
        "offset": offset
    }


def extract_ad_data(ads):
    """Extrait les données importantes des annonces.
    
    Args:
        ads: Liste des annonces brutes depuis l'API
        
    Returns:
        DataFrame pandas avec les données structurées
    """
    return pd.DataFrame([extract_single_ad(ad) for ad in ads])


def extract_single_ad(ad):
    """Extrait les données d'une seule annonce.
    
    Args:
        ad: Dictionnaire contenant les données brutes d'une annonce
        
    Returns:
        Dict avec les données structurées de l'annonce
    """
    info = get_basic_ad_info(ad)
    add_attributes_to_ad(info, ad.get("attributes", []))
    return info


def get_basic_ad_info(ad):
    """Récupère les informations de base d'une annonce.
    
    Args:
        ad: Dictionnaire contenant les données brutes d'une annonce
        
    Returns:
        Dict avec les informations de base
    """
    return {
        "list_id": ad.get("list_id"),
        "subject": ad.get("subject"),
        "price": ad.get("price", [None])[0],
        "first_publication_date": ad.get("first_publication_date"),
        "url": ad.get("url"),
        "city": ad.get("location", {}).get("city"),
        "zipcode": ad.get("location", {}).get("zipcode"),
        "lat": ad.get("location", {}).get("lat"),
        "lng": ad.get("location", {}).get("lng"),
        # Attributs techniques (seront remplis par add_attributes_to_ad)
        "brand": None,
        "model": None,
        "year": None,
        "mileage": None,
        "fuel": None,
        "gearbox": None,
        "horsepower": None,
        "horse_power_din": None
    }


def add_attributes_to_ad(info, attributes):
    """Ajoute les attributs techniques à l'annonce.
    
    Args:
        info: Dict contenant les informations de base de l'annonce (modifié en place)
        attributes: Liste des attributs techniques depuis l'API
    """
    # Mapping entre les clés de l'API et les noms de colonnes
    attribute_mapping = {
        "brand": "brand",
        "model": "model",
        "regdate": "year",
        "mileage": "mileage",
        "fuel": "fuel",
        "gearbox": "gearbox",
        "horsepower": "horsepower",
        "horse_power_din": "horse_power_din"
    }
    
    for attr in attributes:
        key = attr.get("key")
        value = attr.get("value")
        
        # Si la clé est dans notre mapping, on l'ajoute à info
        if key in attribute_mapping:
            info[attribute_mapping[key]] = value
