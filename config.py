"""
Fichier config.py
=================
Ce fichier centralise **toutes les configurations** du projet :
  - organisation des dossiers (raw, processed, metrics…),
  - noms des fichiers CSV,
  - constantes pour l'analyse et la visualisation (couleurs, thèmes Plotly),
  - informations sur la saison UEFA Champions League,
  - paramètres liés au scraping et à l'application Streamlit.

En travaillant ainsi, j'évite de disperser mes paramètres dans le code,
et je facilite la maintenance et l'évolution du projet.
"""

from pathlib import Path
from typing import Dict

# ========= Dossiers du projet =========
"""
Je définis la structure de base pour le stockage des données et résultats :
  - data/raw        → données brutes issues du scraping
  - data/processed  → données nettoyées et préparées pour l'analyse
  - data/metrics    → fichiers contenant les métriques calculées
  - data/visuals    → images, graphiques exportés, etc.
"""
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METRICS_DIR = DATA_DIR / "metrics"
VISUALS_DIR = DATA_DIR / "visuals"

# Je crée les dossiers automatiquement s'ils n'existent pas
for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, METRICS_DIR, VISUALS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ========= Fichiers de base =========
"""
Ici, je nomme tous les fichiers CSV utilisés dans mon projet.
Cela permet de changer un nom facilement sans casser le code.
"""
MATCHES_FILE   = "matches.csv"             # Liste complète des matchs
STANDINGS_FILE = "standings.csv"           # Classements par groupe
PLAYERS_FILE   = "players.csv"             # Données joueurs agrégées
TEAMS_FILE     = "teams.csv"               # Données équipes
STATS_FILE     = "match_stats.csv"         # Statistiques des matchs

# Fichiers pour les données traitées (processed)
PROCESSED_MATCHES   = "processed_matches.csv"
PROCESSED_STANDINGS = "processed_standings.csv"
PROCESSED_PLAYERS   = "processed_players.csv"
PROCESSED_TEAMS     = "processed_teams.csv"

# Fichiers pour les métriques
TEAM_METRICS   = "team_metrics.csv"
PLAYER_METRICS = "player_metrics.csv"
MATCH_METRICS  = "match_metrics.csv"

# Templates pour les fichiers de ranking joueurs (compstats)
PLAYER_RANKING_TEMPLATE     = "player_ranking_{season}.csv"
UCL_PLAYER_RANKING_TEMPLATE = "ucl_player_ranking_{season}.csv"

def csv_player_ranking_candidates(season: int):
    """
    Renvoie les deux chemins possibles pour les fichiers de ranking des joueurs,
    en fonction de la saison.
    """
    return [
        RAW_DIR / PLAYER_RANKING_TEMPLATE.format(season=season),
        RAW_DIR / UCL_PLAYER_RANKING_TEMPLATE.format(season=season),
    ]

# ========= Fonctions utilitaires pour chemins =========
"""
Ces helpers me permettent de générer des chemins de fichiers propres et centralisés.
"""
def csv_path_raw(filename: str) -> Path:       
    return RAW_DIR / filename

def csv_path_processed(filename: str) -> Path: 
    return PROCESSED_DIR / filename

def csv_path_metrics(filename: str) -> Path:   
    return METRICS_DIR / filename

def visuals_path(filename: str) -> Path:       
    return VISUALS_DIR / filename

# ========= Thème graphique (UEFA) =========
"""
Couleurs officielles utilisées pour les graphiques et l'application Streamlit.
"""
UEFA_BASE_URL = "https://www.uefa.com"
UEFA_BASE     = UEFA_BASE_URL

UEFA_BLUE       = "#003366"   # Bleu UEFA
UEFA_LIGHT_BLUE = "#0066cc"   # Bleu clair
GOLD            = "#FFD700"   # Or
WHITE           = "#FFFFFF"   # Blanc
LIGHT_GRAY      = "#F8F9FA"   # Gris clair
DARK_GRAY       = "#343A40"   # Gris foncé

# Thème par défaut pour Plotly
PLOTLY_TEMPLATE = {
    "paper_bgcolor": WHITE,
    "plot_bgcolor": LIGHT_GRAY,
    "font_color": DARK_GRAY,
    "colorway": [UEFA_BLUE, UEFA_LIGHT_BLUE, GOLD, "#8A8D91", "#7FBA00", "#FF8C00"],
}

# ========= Normalisation des phases =========
"""
Je crée un dictionnaire pour uniformiser les noms des phases de la compétition.
Cela me permet de traiter les données sans variations d'écriture (ex: "Quarterfinals" vs "Quarter-finals").
"""
STAGES = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
STAGE_MAP: Dict[str, str] = {
    "Group": "Group Stage", "Groups": "Group Stage", "Group Stage": "Group Stage", "Group stage": "Group Stage",
    "Round of 16": "Round of 16", "R16": "Round of 16",
    "Quarterfinals": "Quarter-finals", "Quarter-finals": "Quarter-finals",
    "Semifinals": "Semi-finals", "Semi-finals": "Semi-finals",
    "Final": "Final",
}

# ========= Informations sur la saison =========
"""
Je stocke ici les informations statiques sur la saison 2023/2024 :
dates, champion, finaliste, stats globales, etc.
Cela peut être utile pour les visuels et pour le rapport final.
"""
COMPETITION_ID = 1
SEASON_YEAR    = 2024  # 2024 → saison 2023/24 (UEFA)

SEASON_INFO = {
    "season": "2023/2024",
    "start_date": "2023-09-19",
    "end_date": "2024-06-01",
    "champion": "Real Madrid",
    "runner_up": "Borussia Dortmund",
    "total_matches": 125,
    "total_teams": 32,
    "total_goals": 327,
    "top_scorer": "Kylian Mbappé",
    "top_scorer_goals": 8,
    "best_player": "Jude Bellingham",
}

# Liste des 32 équipes de la saison
UEFA_TEAMS = [
    "Arsenal","Atlético Madrid","Barcelona","Bayern Munich","Benfica","Braga","Copenhagen","Celtic",
    "Crvena zvezda","Dinamo Zagreb","Feyenoord","Galatasaray","Inter Milan","Lazio","Leipzig","Lens",
    "Manchester City","Manchester United","Milan","Young Boys","Napoli","Newcastle United","Paris Saint-Germain","Porto",
    "PSV","Real Madrid","Real Sociedad","Salzburg","Sevilla","Shakhtar Donetsk","Union Berlin","Antwerp",
]

# ========= Configurations d'exécution =========
"""
Ici, je centralise tous les paramètres pour le scraping, Streamlit et les logs.
"""
# Configuration pour le scraping
SCRAPING_CONFIG = {
    "timeout": 30,
    "retries": 3,
    "delay_between_requests": 0.5,
    "headless": True
}

# Configuration pour l'application Streamlit
STREAMLIT_CONFIG = {
    "page_title": "UCL 2023/2024 Analysis",
    "page_icon": "⚽",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# En cas de données manquantes, utiliser un échantillon ?
USE_SAMPLE_IF_MISSING = False

# Configuration des logs
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "ucl_analysis.log"
}

# ========= Export public (pour import dans d'autres fichiers) =========
__all__ = [
    "RAW_DIR","PROCESSED_DIR","METRICS_DIR","VISUALS_DIR",
    "MATCHES_FILE","STANDINGS_FILE","PLAYERS_FILE","TEAMS_FILE","STATS_FILE",
    "PROCESSED_MATCHES","PROCESSED_STANDINGS","PROCESSED_PLAYERS","PROCESSED_TEAMS",
    "TEAM_METRICS","PLAYER_METRICS","MATCH_METRICS",
    "csv_path_raw","csv_path_processed","csv_path_metrics","visuals_path",
    "PLAYER_RANKING_TEMPLATE","UCL_PLAYER_RANKING_TEMPLATE","csv_player_ranking_candidates",
    "UEFA_BASE_URL","UEFA_BASE","UEFA_BLUE","UEFA_LIGHT_BLUE","GOLD","WHITE","LIGHT_GRAY","DARK_GRAY","PLOTLY_TEMPLATE",
    "STAGES","STAGE_MAP","COMPETITION_ID","SEASON_YEAR","SEASON_INFO","UEFA_TEAMS",
    "SCRAPING_CONFIG","STREAMLIT_CONFIG","LOGGING_CONFIG","USE_SAMPLE_IF_MISSING",
]
