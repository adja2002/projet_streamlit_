# UEFA Champions League 2023/2024 — Analyse & Visualisation

# Tout le code du projet est intégralement commenté pour faciliter la lecture, la maintenance et la réutilisation.

Application **Streamlit** pour **scraper**, **traiter** et **visualiser** la saison **UEFA Champions League 2023/24** : matchs, équipes, joueurs, classements et graphiques interactifs (Plotly).

Installation :
1.executer le requirement.txt ou se trouve toute les bibliotheque necessaires pour notre projet
et executer le requirements.txt en faisant 
2. Installer les dépendances avec : pip install -r requirements.txt
3. Lancer l’application : streamlit run app.py

Structure :
- app.py : fichier principal de l’application.
- scraping/scrap_adja.py : module pour la récupération des données.
- processing/data_processor.py : nettoyage et préparation des données.
- data/ : stockage des données brutes, traitées et des métriques.

Sections de l’application :
- Aperçu.
-Scraper
-processing
- Matchs.
- Equipes.
- Joueurs.
- Histoire.


Si des CSV existent déjà dans data/raw/ (ou si vous laissez les données d’exemple activées) :
streamlit run app/app.py
L’app s’ouvre sur http://localhost:8501/.

# Réponses aux Questions posées dans la consigne 

1) Comment récupérer efficacement toutes ces données historiques ?

Les données ont été récupérées en combinant deux approches complémentaires :

API UEFA : appels aux endpoints match.uefa.com avec pagination et retries pour la robustesse réseau.

Fallback HTML/JSON : quand une information manquait côté API, le JSON __NEXT_DATA__ des pages publiques UEFA a été parsé pour compléter (lineups, évènements, etc.).

Le module scraping/scrap_adja.py orchestre ces flux. Il collecte les 125 matchs, les classements, les lineups, les arbitres, l’affluence, et les événements (buts, passes, cartons). Les données sont stockées en CSV dans data/raw/ pour être ensuite traitées et nettoyées.

2) Comment structurer les données pour que les analyses soient pertinentes ?

La structuration repose sur trois couches clairement séparées :

RAW (data/raw/) : données brutes issues du scraping (sans modification).

PROCESSED (data/processed/) : données nettoyées et typées, prêtes pour l’analyse.

METRICS (data/metrics/) : indicateurs agrégés (points par match, performances joueur/équipe, etc.).

Des clés uniques (match_id, player_id, team_id) et une normalisation des noms d’équipes/joueurs assurent la cohérence, les jointures et les filtres.

3) Quelles visualisations racontent le mieux l’histoire de la compétition ?

Le tableau de bord est organisé selon les pages de l’interface ci-dessous — avec les noms exacts utilisés dans l’app :

Aperçu : KPIs globaux (Matches, Goals, Attendance, Teams), timeline des matchs, distributions (buts, résultats).

Scraper : boutons pour récupérer/mettre à jour les matchs, groupes, team stats, et joueurs par match.

Processing : lancement du pipeline de nettoyage et d’agrégation (génération de processed/*.csv et metrics/*.csv) + contrôles rapides de qualité.

Matchs : heatmap des résultats par adversaire, boxplots domicile/extérieur, répartition H/D/A.

Équipes : radars d’équipes, scatter “style de jeu”, standings de groupes, focus Seulement KO.

Joueurs : top buteurs/passeurs, minutes vs buts, radars individuels, goal involvement.

Histoires : bracket des phases finales, parcours d’équipes (ex. Real Madrid), surprises et sunburst des buts.

Ces pages permettent de passer du global au détail et de raconter l’histoire de la saison de façon progressive et visuelle.

4) Comment organiser l’interface Streamlit pour une exploration fluide ?

Navigation claire (sticky) : Aperçu, Scraper, Processing, Matchs, Équipes, Joueurs, Histoires.

Filtres globaux persistants : phase, équipe(s), “Seulement KO”, mémorisés dans st.session_state.

Thème & design : thème sombre homogène, composants visuels uniformes, largeur adaptative (width='stretch').

Flux logique recommandé :

Scraper les données →

Processing (nettoyage + métriques) →

Explorer Aperçu, Matchs, Équipes, Joueurs et Histoires.

Ces choix garantissent une expérience intuitive, des chargements rapides (cache Streamlit) et une analyse sans friction.
# quelques commandes utils
Matchs (liste + méta, avec enrichissement):python scraping/scrap_adja.py matches --limit 200 --season-year 2024
Groupes / Classements:python scraping/scrap_adja.py groups --season-year 2024

Joueurs:streamlit -m scraping.scrap_adja players_fast(plus rapide)

Tout le pipeline (processed + metrics):python data_processor.py












