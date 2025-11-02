"""API simple pour synchroniser les annonces automobiles vers Google Sheets."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from leboncoin_client import get_leboncoin_ads
from annonces_client import run_annonces_scraping
from google_sheets import save_to_sheet, get_good_deals
from training import train_model_for_source
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Watch API Simple", version="1.0.0")

# Configuration CORS - Permet les requêtes depuis n'importe quelle origine
# Note : En production, il est recommandé de restreindre allow_origins aux domaines autorisés
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Endpoint racine - Retourne les informations de base de l'API."""
    return {
        "message": "Car Watch API Simple", 
        "endpoints": [
            "/update_leboncoin", 
            "/update_annoncesautomobile",
            "/good_deals",
            "/train"
        ]
    }


@app.post("/update_leboncoin")
def update_leboncoin(limit: int = 100):
    """Synchronise les annonces LeBonCoin vers Google Sheets.
    
    Args:
        limit: Nombre maximum d'annonces à récupérer (défaut: 100)
        
    Returns:
        Dict avec le statut de succès et le nombre d'annonces récupérées
    """
    try:
        logger.info(f"Début de la synchronisation LeBonCoin (limit={limit})")
        data = get_leboncoin_ads(limit)
        save_to_sheet(data, "leboncoin_raw", "url", source="leboncoin")
        logger.info(f"Synchronisation LeBonCoin terminée avec succès ({len(data)} annonces)")
        return {"success": True, "count": len(data)}
    except Exception as e:
        logger.error(f"Erreur lors de la synchronisation LeBonCoin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_annoncesautomobile")
def update_annonces(pages: int = 1):
    """Synchronise les annonces automobiles vers Google Sheets.
    
    Args:
        pages: Nombre de pages à scraper (défaut: 1)
        
    Returns:
        Dict avec le statut de succès et le nombre d'annonces récupérées
    """
    try:
        logger.info(f"Début de la synchronisation Annonces-Automobile (pages={pages})")
        data = run_annonces_scraping(pages)
        save_to_sheet(data, "annonces-automobiles_raw", "id_annonce",source="annonces")
        logger.info(f"Synchronisation Annonces-Automobile terminée avec succès ({len(data)} annonces)")
        return {"success": True, "count": len(data)}
    except Exception as e:
        logger.error(f"Erreur lors de la synchronisation Annonces-Automobile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/good_deals")
def get_good_deals_endpoint(
    source: str = "leboncoin_raw", 
    discount: float = 0.8,
    km_max: int = None,
    start_date: str = None,
    end_date: str = None
):
    """Récupère les véhicules avec un prix en dessous du prix prédit.
    
    Args:
        source: Nom du worksheet ("leboncoin_raw" ou "annonces-automobiles_raw")
        discount: Seuil de remise (0.8 = 20% de remise, 0.7 = 30% de remise)
        km_max: Kilométrage maximum (optionnel)
        start_date: Date de début au format DD/MM/YYYY (optionnel)
        end_date: Date de fin au format DD/MM/YYYY (optionnel)
        
    Returns:
        Dict avec le statut, le nombre de bonnes affaires et la liste des véhicules
    """
    try:
        logger.info(f"Recherche de bonnes affaires (source={source}, discount={discount})")
        deals = get_good_deals(source, discount, km_max, start_date, end_date)
        logger.info(f"Recherche terminée: {len(deals)} bonnes affaires trouvées")
        return {
            "success": True, 
            "count": len(deals),
            "filters": {
                "source": source,
                "discount_threshold": discount,
                "km_max": km_max,
                "start_date": start_date,
                "end_date": end_date
            },
            "vehicles": deals
        }
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de bonnes affaires: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_model_endpoint(source: str):
    """Entraîne un modèle de prédiction de prix pour une source donnée.
    
    Cette route permet de déclencher l'entraînement d'un modèle directement via l'API.
    Les données sont récupérées depuis Google Sheets, le modèle est entraîné et sauvegardé
    automatiquement.
    
    Args:
        source: Source des données ('leboncoin' ou 'annonces')
        
    Returns:
        Dict avec le statut de succès, les métriques et les fichiers générés
        
    Example:
        POST /train?source=leboncoin
        POST /train?source=annonces
    """
    try:
        logger.info(f"Début de l'entraînement du modèle pour la source: {source}")
        
        # Validation de la source
        if source not in ['leboncoin', 'annonces']:
            raise HTTPException(
                status_code=400, 
                detail=f"Source invalide: {source}. Doit être 'leboncoin' ou 'annonces'"
            )
        
        # Appel de la fonction d'entraînement
        result = train_model_for_source(source)
        
        logger.info(f"Entraînement terminé avec succès pour {source}")
        logger.info(f"  - Échantillons d'entraînement: {result['training_samples']}")
        logger.info(f"  - R²: {result['metrics']['R2']:.4f}")
        logger.info(f"  - RMSE: {result['metrics']['RMSE']:.2f} €")
        logger.info(f"  - MAE: {result['metrics']['MAE']:.2f} €")
        
        return result
        
    except ValueError as e:
        # Erreur de validation (source invalide)
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Autres erreurs (Google Sheets, entraînement, etc.)
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Démarrage de l'API sur http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)