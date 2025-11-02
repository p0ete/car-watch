"""Configuration simple pour l'API d'annonces automobiles."""

import os

# Configuration Google Sheets
# SPREADSHEET_ID: Identifiant unique du Google Sheet (visible dans l'URL)
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")

# GOOGLE_SHEETS_CREDENTIALS: Contenu JSON des credentials du service account Google
# Format attendu: chaîne JSON complète avec les clés privées
GOOGLE_SHEETS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

# Configuration LeBonCoin API (via RapidAPI)
# URL de l'API LeBonCoin hébergée sur RapidAPI
LEBONCOIN_API_URL = "https://leboncoin13.p.rapidapi.com/api/v1/listings/search"

# Headers nécessaires pour l'authentification RapidAPI
LEBONCOIN_HEADERS = {
    "X-Rapidapi-Key": os.getenv("X_RAPIDAPI_KEY", ""),
    "X-Rapidapi-Host": "leboncoin13.p.rapidapi.com",
    "Content-Type": "application/json"
}

# Configuration AnnoncesAutomobiles
# URL de base pour le scraping avec filtres pré-appliqués
# k=150000 : kilométrage max, a=2012-0000 : année min
ANNONCES_URL = "https://www.annonces-automobile.com/l-s/occasion?k=150000&a=2012-0000"

# Configuration de la prédiction de prix
# ENABLE_PREDICTION: Active/désactive l'ajout automatique de la colonne predicted_price
# True = la prédiction est activée, False = désactivée
ENABLE_PREDICTION = os.getenv("ENABLE_PREDICTION", "True").lower() == "true"