"""Client simple pour scraper les annonces automobiles - version améliorée."""

import asyncio
import random
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
from config import ANNONCES_URL
import logging 

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_price(price_str):
    """Nettoie et convertit le prix en int.
    
    Gère différents formats de prix:
    - "33.900 €" → 33900
    - "31.990 €" → 31990
    - "7 990 €" → 7990
    - "7990€" → 7990
    
    Args:
        price_str: Chaîne de caractères contenant le prix
        
    Returns:
        Prix en entier ou None si la conversion échoue
    """
    if not price_str:
        return None
    
    try:
        # Enlever tous les caractères non numériques sauf point, virgule et espace
        cleaned = re.sub(r'[^\d.,\s]', '', price_str).strip()
        
        # Remplacer les espaces par rien (séparateurs de milliers)
        cleaned = cleaned.replace(' ', '')
        
        # Gestion des points (séparateurs de milliers vs décimales)
        if '.' in cleaned:
            parts = cleaned.split('.')
            
            if len(parts) == 2:
                # Si la partie après le point a 3 chiffres, c'est un séparateur de milliers
                if len(parts[-1]) == 3:
                    cleaned = cleaned.replace('.', '')
            elif len(parts) > 2:
                # Plusieurs points = séparateurs de milliers
                if all(len(p) == 3 for p in parts[1:]):
                    cleaned = cleaned.replace('.', '')
        
        # Si on a une virgule, la remplacer par un point pour la conversion
        if ',' in cleaned:
            cleaned = cleaned.replace(',', '.')
        
        # Convertir en float puis int
        float_val = float(cleaned)
        result = int(float_val)        
        return result
        
    except ValueError as e:
        logger.error(f"Erreur de conversion du prix '{price_str}': {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la conversion du prix '{price_str}': {e}")
        return None


def clean_kilometrage(km_str):
    """Nettoie et convertit le kilométrage en int.
    
    Gère différents formats:
    - "125.000km" → 125000
    - "125 000 km" → 125000
    - "125000" → 125000
    
    Args:
        km_str: Chaîne de caractères contenant le kilométrage
        
    Returns:
        Kilométrage en entier ou None si la conversion échoue
    """
    if not km_str:
        return None
    
    try:
        # Enlever tous les caractères non numériques
        cleaned = re.sub(r'[^\d]', '', km_str)
        # Convertir en int
        return int(cleaned) if cleaned else None
    except ValueError:
        logger.error(f"Erreur de conversion du kilométrage '{km_str}'")
        return None


async def get_annonces_ads(max_pages=1):
    """Récupère les annonces depuis annonces-automobile.com.
    
    Args:
        max_pages: Nombre de pages à scraper
        
    Returns:
        DataFrame pandas contenant les annonces extraites
    """
    logger.info(f"Lancement du scraping avec playwright (max_pages={max_pages})...")
    
    async with async_playwright() as p:
        # Lancement du navigateur en mode headless
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()
        all_ads = []
        
        # Parcours de chaque page
        for page_num in range(1, max_pages + 1):
            logger.info(f"Analyse de la page {page_num}/{max_pages}")
            ads_on_page = await scrape_single_page(page, page_num)
            logger.info(f"{len(ads_on_page)} annonces récupérées.")
            all_ads.extend(ads_on_page)
            
            # Pause aléatoire entre les pages pour simuler un comportement humain
            await asyncio.sleep(random.uniform(2.0, 4.0))
        
        await browser.close()
        logger.info(f"Au total {len(all_ads)} annonces ont été scrapées.")
        return pd.DataFrame(all_ads)


async def scrape_single_page(page, page_num):
    """Scrape une seule page d'annonces.
    
    Args:
        page: Objet Page de Playwright
        page_num: Numéro de la page à scraper
        
    Returns:
        Liste de dictionnaires contenant les données des annonces
    """
    url = build_pagination_url(page_num)
    
    try:
        await page.goto(url, timeout=60000)
        # Attente aléatoire pour simuler un comportement humain
        await page.wait_for_timeout(random.randint(2000, 4000))
    except PlaywrightTimeoutError:
        logger.error(f"Timeout lors du chargement de la page {page_num}")
        return []
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la page {page_num}: {e}")
        return []
    
    # Essayer plusieurs sélecteurs pour trouver les cartes d'annonces
    cards = await find_cards_with_selectors(page)
    
    if not cards:
        logger.warning(f"Aucune carte d'annonce trouvée sur la page {page_num}")
        return []
    
    # Extraction des données de chaque carte
    ads_data = []
    for card in cards:
        ad_info = await extract_annonce_fields(card)
        # Ne garder que les annonces avec un ID valide
        if ad_info and ad_info.get('id_annonce'):
            ads_data.append(ad_info)
    
    return ads_data


def build_pagination_url(page_number):
    """Construit l'URL pour une page spécifique.
    
    Args:
        page_number: Numéro de la page
        
    Returns:
        URL complète avec le paramètre de pagination
    """
    url_parts = list(urlparse(ANNONCES_URL))
    query = parse_qs(url_parts[4])
    query['pg'] = [str(page_number)]
    url_parts[4] = urlencode(query, doseq=True)
    return urlunparse(url_parts)


async def find_cards_with_selectors(page):
    """Trouve les cartes d'annonces avec plusieurs sélecteurs.
    
    Essaie plusieurs sélecteurs CSS pour s'adapter aux changements de structure HTML.
    
    Args:
        page: Objet Page de Playwright
        
    Returns:
        Liste des éléments trouvés ou liste vide
    """
    selectors = [
        'div.annlisting',
        'div.listing', 
        'div.annonce',
        'div[class*="ann"]',
        '.card',
        'article'
    ]
    
    for selector in selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            cards = await page.query_selector_all(selector)
            if cards:
                logger.info(f"Cartes trouvées avec le sélecteur: {selector}")
                return cards
        except PlaywrightTimeoutError:
            continue
        except Exception as e:
            logger.debug(f"Erreur avec le sélecteur {selector}: {e}")
            continue
    
    return []


async def extract_annonce_fields(card):
    """Extrait tous les champs d'une carte d'annonce.
    
    Args:
        card: Élément DOM de la carte d'annonce
        
    Returns:
        Dict contenant les données de l'annonce ou None en cas d'erreur
    """
    try:
        # Initialisation de la structure de données
        annonce = {
            'titre': await safe_text(card, 'span.fw-normal'),
            'sous_titre': await safe_text(card, 'span.crop-text-2.fw-bold'),
            'categorie': await safe_text(card, 'div.badge.border-danger'),
            'prix': await extract_price(card),
            'url': '',
            'id_annonce': '',
            'kilometrage': None,
            'annee': None,
            'boite': '',
            'energie': '',
            'couleur': '',
            'localisation': ''
        }
        
        # Extraction de l'URL et de l'ID
        url, id_annonce = await extract_url_and_id(card)
        annonce['url'] = url
        annonce['id_annonce'] = id_annonce
        
        # Extraction des champs techniques par icône
        await extract_technical_fields(card, annonce)
        
        # Extraction de la localisation
        annonce['localisation'] = await extract_location(card)
        
        return annonce
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction d'une annonce: {e}")
        return None


async def extract_price(card):
    """Extrait le prix de l'annonce et le convertit en int.
    
    Args:
        card: Élément DOM de la carte d'annonce
        
    Returns:
        Prix en entier ou None
    """
    prix_div = await card.query_selector('div.w-100.text-primary.h4.mb-0')
    if prix_div:
        txt = await prix_div.inner_text()
        txt = re.sub(r'\s+', ' ', txt).strip()
        return clean_price(txt)
    return None


async def extract_url_and_id(card):
    """Extrait l'URL et l'ID de l'annonce.
    
    Args:
        card: Élément DOM de la carte d'annonce
        
    Returns:
        Tuple (url, id_annonce)
    """
    a = await card.query_selector('a.stretched-link')
    if a:
        href = await a.get_attribute('href')
        if href:
            url = f"https://www.annonces-automobile.com{href}"
            # Extraction de l'ID depuis l'URL (format: /123456)
            match = re.search(r'/(\d+)', href)
            id_annonce = match.group(1) if match else ""
            return url, id_annonce
    return "", ""


async def extract_technical_fields(card, annonce):
    """Extrait les champs techniques par icône.
    
    Args:
        card: Élément DOM de la carte d'annonce
        annonce: Dict contenant les données de l'annonce (modifié en place)
    """
    # Mapping entre les icônes Font Awesome et les champs de l'annonce
    icon_mapping = {
        'fa-gauge': 'kilometrage',
        'fa-calendar': 'annee',
        'fa-gears': 'boite',
        'fa-gas-pump': 'energie',
        'fa-palette': 'couleur'
    }
    
    for icon, field in icon_mapping.items():
        span = await card.query_selector(f'i.{icon}')
        if span:
            try:
                # Récupération du texte du parent de l'icône
                valeur = await span.evaluate('node => node.parentElement.textContent')
                if valeur:
                    valeur = valeur.strip()
                    # Traitement spécial pour le kilométrage
                    if field == 'kilometrage':
                        annonce[field] = clean_kilometrage(valeur)
                    else:
                        annonce[field] = valeur
                else:
                    # Valeur par défaut selon le type de champ
                    annonce[field] = None if field in ['kilometrage', 'annee'] else ""
            except Exception as e:
                logger.debug(f"Erreur lors de l'extraction du champ {field}: {e}")
                annonce[field] = None if field in ['kilometrage', 'annee'] else ""


async def extract_location(card):
    """Extrait la localisation de l'annonce.
    
    Args:
        card: Élément DOM de la carte d'annonce
        
    Returns:
        Chaîne de caractères contenant la localisation ou chaîne vide
    """
    block = await card.query_selector('div.mt-2.pt-2.fs-13.text-muted')
    if block:
        try:
            block_text = await block.inner_text()
            block_text = block_text.replace('\n', ' ').strip()
            
            # Essayer de trouver la localisation après un tiret
            m = re.search(r'-\s*(.*)$', block_text)
            localisation = m.group(1).strip() if m else ""
            
            # Si pas trouvé, essayer un autre pattern
            if not localisation:
                m = re.search(r'(\w+, \(\d+\) [\w\- ]+)$', block_text)
                localisation = m.group(1).strip() if m else ""
            
            return localisation
        except Exception as e:
            logger.debug(f"Erreur lors de l'extraction de la localisation: {e}")
            pass
    return ""


async def safe_text(parent, selector):
    """Retourne le texte d'un élément ou chaîne vide.
    
    Args:
        parent: Élément DOM parent
        selector: Sélecteur CSS de l'élément à extraire
        
    Returns:
        Texte de l'élément ou chaîne vide
    """
    try:
        element = await parent.query_selector(selector)
        if element:
            text = await element.inner_text()
            return text.strip() if text else ""
    except Exception:
        pass
    return ""


def run_annonces_scraping(max_pages=1):
    """Lance le scraping de manière synchrone.
    
    Args:
        max_pages: Nombre de pages à scraper
        
    Returns:
        DataFrame pandas contenant les annonces extraites
    """
    return asyncio.run(get_annonces_ads(max_pages))
