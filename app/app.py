# ================================================
# UCL 23/24 — Dashboard Streamlit 
# ================================================
# Dans ce fichier, je construis une application Streamlit
# pour explorer la saison UCL 2023/2024 : scraping, processing,
# métriques et visualisations (matchs, équipes, joueurs, histoires).
# J’ai pris soin de séparer proprement :
#   - le chargement des données (raw/processed/metrics)
#   - des utilitaires d’affichage (dataframe, plots, CSS, navigation)
#   - des visuels “ad hoc” spécifiques aux analyses de phase, surprises, difficulté de parcours
#   - la navigation multi-pages (aperccu, Scraper, Processing, etc.)
#
# Remarque importante pour l’évaluateur : j’ai anticipé les colonnes manquantes
# et les CSV vides (retours réseau, première exécution, etc.) pour éviter les crashs.
# ================================================

# --- make project root importable ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# J’importe la configuration centrale (couleurs, chemins, noms de fichiers)
from config import (
    UEFA_BLUE, UEFA_LIGHT_BLUE, GOLD, WHITE,
    csv_path_raw, csv_path_processed, csv_path_metrics,
    MATCHES_FILE, STANDINGS_FILE, PLAYERS_FILE, TEAMS_FILE,
    PROCESSED_MATCHES, PROCESSED_STANDINGS, PROCESSED_PLAYERS, PROCESSED_TEAMS,
    TEAM_METRICS, PLAYER_METRICS, MATCH_METRICS,
)

# J’importe les fonctions de scraping (matchs & groupes) + mon processeur de données + le module de visualisations
from scraping.scrap_adja import scrape_matches, scrape_groups
from data_processor import DataProcessor
from viz import UCLVisualizer

# ====================================================
# ===============  Helpers d’Entrée/Sortie  ==========
# ====================================================
@st.cache_data(show_spinner=False)
def _read_csv(pathlike) -> pd.DataFrame:
    """
    Lecture sécurisée d’un CSV. Je cache le résultat pour accélérer les re-runs.
    - Si le fichier n’existe pas ou est vide => je renvoie un DataFrame vide.
    """
    try:
        path = str(pathlike)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path, encoding="utf-8")
    except Exception:
        # Je protège l’UI contre toute exception d’E/S
        pass
    return pd.DataFrame()

def load_all():
    """
    Je centralise tous les chargements de fichiers :
      - 'raw'  : bruts issus du scraping
      - 'proc' : tables nettoyées et normalisées
      - 'metrics' : agrégats calculés pour équipes/joueurs/matchs
    Je renvoie un dict structuré pour simplifier la suite.
    """
    # RAW
    raw_matches   = _read_csv(csv_path_raw(MATCHES_FILE))
    raw_standings = _read_csv(csv_path_raw(STANDINGS_FILE))
    raw_players   = _read_csv(csv_path_raw(PLAYERS_FILE))
    raw_teams     = _read_csv(csv_path_raw(TEAMS_FILE))

    # PROCESSED
    proc_matches  = _read_csv(csv_path_processed(PROCESSED_MATCHES))
    proc_stand    = _read_csv(csv_path_processed(PROCESSED_STANDINGS))
    proc_players  = _read_csv(csv_path_processed(PROCESSED_PLAYERS))
    proc_teams    = _read_csv(csv_path_processed(PROCESSED_TEAMS))

    # METRICS
    team_metrics  = _read_csv(csv_path_metrics(TEAM_METRICS))
    player_metrics= _read_csv(csv_path_metrics(PLAYER_METRICS))
    match_metrics = _read_csv(csv_path_metrics(MATCH_METRICS))

    return {
        "raw": dict(matches=raw_matches, standings=raw_standings, players=raw_players, teams=raw_teams),
        "proc": dict(matches=proc_matches, standings=proc_stand, players=proc_players, teams=proc_teams),
        "metrics": dict(teams=team_metrics, players=player_metrics, matches=match_metrics),
    }

# ====================================================
# ===============  Utilitaires d’affichage ===========
# ====================================================
def _tidy_for_display(df: pd.DataFrame, maybe_cols=("group","country","team_id")) -> pd.DataFrame:
    """
    Petites retouches d’affichage :
      - je supprime les colonnes remplies de vides ('', None, NaN…)
      - sinon je remplace les valeurs “textuelles vides” par ''.
    Mon but : rendre les tableaux plus lisibles dans Streamlit.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in maybe_cols:
        if c in out.columns:
            is_blank = out[c].isna() | out[c].astype(str).str.strip().isin(["", "None", "none", "NaN", "nan", "Null", "null"])
            if bool(is_blank.all()):
                out = out.drop(columns=[c])
            else:
                out[c] = out[c].where(~is_blank, "")
    return out

# ====================================================
# =================  Visuels ad hoc  =================
# ====================================================
def _phase_from_stage(stage: str) -> str:
    """
    Je mappe les libellés ‘stage’ (bruts) vers des phases lisibles et ordonnées.
    Ce standard m’aide à grouper/ordonner correctement les graphiques.
    """
    if not isinstance(stage, str):
        return "Unknown"
    s = stage.lower()
    if "group" in s: return "Group Stage"
    if "quarter" in s: return "Quarter-finals"
    if "semi" in s: return "Semi-finals"
    if "final" in s and "semi" not in s: return "Final"
    if "round" in s: return "Round of 16"
    return stage

def fig_phase_evolution(matches_df: pd.DataFrame):
    """
    Graphique : moyenne de buts par phase + nombre de matchs (en label).
    - Je calcule total_goals = home_goals + away_goals
    - Je groupe par phase
    - Je force l’ordre : Groups -> R16 -> QF -> SF -> Final
    """
    if matches_df is None or matches_df.empty:
        return None
    df = matches_df.copy()
    need = {"home_goals","away_goals","stage"}
    if not need.issubset(df.columns): return None

    df["phase"] = df["stage"].apply(_phase_from_stage)
    df["total_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0) + \
                        pd.to_numeric(df["away_goals"], errors="coerce").fillna(0)
    g = df.groupby("phase", as_index=False).agg(matches=("match_id","count"),
                                                avg_goals=("total_goals","mean"))
    order = ["Group Stage","Round of 16","Quarter-finals","Semi-finals","Final"]
    g["phase"] = pd.Categorical(g["phase"], categories=order, ordered=True)
    g = g.sort_values("phase")

    fig = px.bar(
        g, x="phase", y="avg_goals", text=g["matches"],
        labels={"phase":"Phase","avg_goals":"Average Goals"},
        title="Average Goals per Phase (with match counts)"
    )
    # J’uniformise la charte (couleur, fond transparent — ensuite je passe au thème sombre global)
    fig.update_traces(marker_color=UEFA_LIGHT_BLUE, texttemplate="%{text} matches")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def fig_upsets(matches_df: pd.DataFrame, team_metrics: pd.DataFrame, top_n=12):
    """
    Graphique : ‘surprises’ de la saison.
    Idée : je prends le classement PPG des équipes (rank), et pour chaque match je mesure
    l’écart de rang entre perdant et gagnant => plus l’écart est grand, plus la surprise est forte.
    """
    if matches_df is None or team_metrics is None or matches_df.empty or team_metrics.empty:
        return None
    if not {"team","points_per_game"}.issubset(team_metrics.columns):
        return None

    # Classement des équipes selon 'points_per_game'
    rk = team_metrics[["team","points_per_game"]].sort_values("points_per_game", ascending=False).reset_index(drop=True)
    rk["rank"] = np.arange(1, len(rk)+1)

    # Préparation des matchs avec rangs des équipes
    m = matches_df.copy()
    need = {"home_team","away_team","home_goals","away_goals","stage"}
    if not need.issubset(m.columns): return None

    m = m.merge(rk.rename(columns={"team":"home_team","rank":"home_rank"}), on="home_team", how="left")
    m = m.merge(rk.rename(columns={"team":"away_team","rank":"away_rank"}), on="away_team", how="left")

    # Détermination du vainqueur/perdant (match sec)
    m["winner"] = np.where(m["home_goals"]>m["away_goals"], m["home_team"],
                    np.where(m["away_goals"]>m["home_goals"], m["away_team"], np.nan))
    m["loser"]  = np.where(m["home_goals"]<m["away_goals"], m["home_team"],
                    np.where(m["away_goals"]<m["home_goals"], m["away_team"], np.nan))
    # Upset = différence de rang (perdant - gagnant)
    m["winner_rank"] = np.where(m["winner"].eq(m["home_team"]), m["home_rank"], m["away_rank"])
    m["loser_rank"]  = np.where(m["loser"].eq(m["home_team"]), m["home_rank"], m["away_rank"])
    m["upset_score"] = (m["loser_rank"] - m["winner_rank"]).fillna(0)

    upsets = m[m["upset_score"]>0].copy()
    if upsets.empty: return None
    upsets["label"] = upsets.apply(lambda r: f"{r['winner']} beat {r['loser']} ({int(r['home_goals'])}-{int(r['away_goals'])})", axis=1)
    upsets["phase"] = upsets["stage"].apply(_phase_from_stage)

    top = upsets.sort_values(["upset_score","phase"], ascending=[False, True]).head(top_n)
    fig = px.bar(top, x="upset_score", y="label", color="phase", orientation="h",
                 title="Surprises de la saison (ranking-based upsets)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def fig_path_difficulty(matches_df: pd.DataFrame, team_metrics: pd.DataFrame, only_knockouts=True):
    """
    Graphique : difficulté moyenne du parcours d’une équipe (moyenne des PPG des adversaires).
    - Je passe chaque match en format long (team/opponent/phase)
    - Je joins la métrique 'points_per_game' de l’adversaire
    - Je moyenne par équipe (option : seulement phases à élimination directe)
    """
    if matches_df is None or team_metrics is None or matches_df.empty or team_metrics.empty:
        return None
    if not {"team","points_per_game"}.issubset(team_metrics.columns):
        return None

    m = matches_df.copy()
    need = {"home_team","away_team","stage"}
    if not need.issubset(m.columns): return None
    m["phase"] = m["stage"].apply(_phase_from_stage)
    if only_knockouts:
        m = m[m["phase"].isin(["Round of 16","Quarter-finals","Semi-finals","Final"])]

    # Je “déplie” chaque match pour attribuer l’adversaire à chaque équipe
    long = []
    for _, r in m.iterrows():
        long.append({"team": r["home_team"], "opponent": r["away_team"], "phase": r["phase"]})
        long.append({"team": r["away_team"], "opponent": r["home_team"], "phase": r["phase"]})
    long = pd.DataFrame(long)
    if long.empty: return None

    ppg = team_metrics[["team","points_per_game"]]
    long = long.merge(ppg.rename(columns={"team":"opponent","points_per_game":"opp_ppg"}), on="opponent", how="left")

    # Agrégat final
    agg = long.groupby("team", as_index=False).agg(path_difficulty=("opp_ppg","mean"))
    agg = agg.sort_values("path_difficulty", ascending=False).head(16)

    fig = px.bar(agg, x="path_difficulty", y="team", orientation="h",
                 title="Parcours le plus difficile (moyenne PPG des adversaires, KO)")
    fig.update_traces(marker_color=UEFA_LIGHT_BLUE)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ====================================================
# ====== Bracket : agrégation des doubles confront. ==
# ====================================================
def build_bracket_data(matches_df: pd.DataFrame) -> dict:
    """
    Je reconstruis un ‘bracket’ à partir des matchs :
      - R16, QF, SF : double confrontation => j’additionne les buts
      - Finale : match sec
    - En cas d’égalité parfaite, je fais un tie-break pragmatique :
      * si l’équipe apparaît dans la phase suivante, je l’utilise comme proxy du vainqueur
      * sinon je prends le dernier match (score) ou un fallback alphabétique
    Objectif : alimenter un visuel ‘flow’ de tournoi (create_tournament_bracket).
    """
    if matches_df is None or matches_df.empty:
        return {}

    need = {"home_team","away_team","home_goals","away_goals","stage","date_utc"}
    if not need.issubset(matches_df.columns):
        return {}

    df = matches_df.copy()
    df["phase"] = df["stage"].apply(_phase_from_stage)

    # Je normalise la date pour trier les manches
    try:
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    except Exception:
        df["date_utc"] = pd.NaT

    phases = ["Round of 16","Quarter-finals","Semi-finals","Final"]
    df = df[df["phase"].isin(phases)].copy()

    next_phase = {"Round of 16": "Quarter-finals", "Quarter-finals": "Semi-finals", "Semi-finals": "Final"}

    data = {"round_of_16": [], "quarter_finals": [], "semi_finals": [], "final": []}
    map_key = {"Round of 16":"round_of_16","Quarter-finals":"quarter_finals","Semi-finals":"semi_finals","Final":"final"}

    # Je pré-calcule les équipes présentes à chaque phase pour aider au tie-break
    teams_by_phase = {
        ph: set(pd.unique(pd.concat([df.loc[df["phase"]==ph, "home_team"],
                                     df.loc[df["phase"]==ph, "away_team"]], ignore_index=True).dropna()))
        for ph in phases
    }

    # Finale (match sec)
    for _, r in df[df["phase"]=="Final"].iterrows():
        data["final"].append({
            "home": r["home_team"], "away": r["away_team"],
            "home_score": int(r["home_goals"]), "away_score": int(r["away_goals"])
        })

    # R16, QF, SF (aller/retour)
    for ph in ["Round of 16","Quarter-finals","Semi-finals"]:
        sub = df[df["phase"]==ph].copy()
        if sub.empty:
            continue

        # Je regroupe les paires (peu importe l’ordre) via frozenset
        sub["pair_key"] = sub.apply(lambda s: frozenset([s["home_team"], s["away_team"]]), axis=1)

        for pk, grp in sub.groupby("pair_key"):
            if len(pk) != 2:
                continue
            t1, t2 = list(pk)

            # Total de buts marqués par t1 et t2 sur les deux manches
            t1_goals = int(pd.to_numeric(grp.loc[grp["home_team"]==t1, "home_goals"], errors="coerce").fillna(0).sum() +
                           pd.to_numeric(grp.loc[grp["away_team"]==t1, "away_goals"], errors="coerce").fillna(0).sum())
            t2_goals = int(pd.to_numeric(grp.loc[grp["home_team"]==t2, "home_goals"], errors="coerce").fillna(0).sum() +
                           pd.to_numeric(grp.loc[grp["away_team"]==t2, "away_goals"], errors="coerce").fillna(0).sum())

            # Je trie par date pour retrouver l’ordre aller/retour
            grp_sorted = grp.sort_values("date_utc")

            # Je choisis une orientation stable (home/away) pour l’affichage agrégé
            if not grp_sorted.empty and pd.notna(grp_sorted.iloc[0]["home_team"]):
                home_agg = grp_sorted.iloc[0]["home_team"]
                away_agg = list(pk - {home_agg})[0]
            else:
                # fallback déterministe
                home_agg, away_agg = sorted(list(pk))

            # Détermination du vainqueur (tie-breaks progressifs)
            if t1_goals > t2_goals:
                winner = t1
            elif t2_goals > t1_goals:
                winner = t2
            else:
                nxt = next_phase.get(ph)
                if nxt and nxt in teams_by_phase:
                    if t1 in teams_by_phase[nxt] and t2 not in teams_by_phase[nxt]:
                        winner = t1
                    elif t2 in teams_by_phase[nxt] and t1 not in teams_by_phase[nxt]:
                        winner = t2
                    else:
                        # Dernier match comme proxy
                        last = grp_sorted.iloc[-1]
                        if last["home_goals"] > last["away_goals"]:
                            winner = last["home_team"]
                        elif last["away_goals"] > last["home_goals"]:
                            winner = last["away_team"]
                        else:
                            # Égalité totale → fallback alphabétique
                            winner = sorted([t1, t2])[0]
                else:
                    winner = sorted([t1, t2])[0]

            # Scores “orientés” selon l’agrégation choisie
            if home_agg == t1:
                home_score, away_score = t1_goals, t2_goals
            else:
                home_score, away_score = t2_goals, t1_goals

            data[map_key[ph]].append({
                "home": home_agg, "away": away_agg,
                "home_score": int(home_score), "away_score": int(away_score),
                "winner": winner
            })

    return data

# ====================================================
# =========== Thème sombre + header + nav ============
# ====================================================
st.set_page_config(
    page_title="UCL 23/24 — Et si on revivait la saison 2023-2024 ?",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Je définis un template Plotly sombre cohérent avec l’identité visuelle
pio.templates["ucl_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#EAF2FF"),
        hoverlabel=dict(bgcolor="#0f1a3f", font=dict(color="#EAF2FF")),
        colorway=["#47b5ff","#7c4dff","#ffd166","#00e5a8","#ff7ab6","#a0e418"],
        xaxis=dict(gridcolor="#222", zerolinecolor="#222", linecolor="#333"),
        yaxis=dict(gridcolor="#222", zerolinecolor="#222", linecolor="#333"),
    )
)
pio.templates.default = "ucl_dark"

def _darken(fig):
    """Petite fonction utilitaire pour harmoniser le thème sombre sur chaque figure."""
    if fig is None:
        return None
    fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000", font=dict(color="#EAF2FF"))
    return fig

# Helpers Streamlit pour gérer le changement d’API : width='stretch' (fallback si version antérieure)
def _p(fig):
    """Affiche un graphique Plotly en occupant toute la largeur (avec fallback)."""
    if fig is not None:
        try:
            st.plotly_chart(_darken(fig), width="stretch", config={"displayModeBar": False})
        except TypeError:
            st.plotly_chart(_darken(fig), use_container_width=True, config={"displayModeBar": False})

def _df(df, **kwargs):
    """Affiche un DataFrame en plein écran (avec fallback selon version Streamlit)."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)

def _inject_css_dark():
    """
    Ici j’injecte mon CSS pour :
      - le bandeau ‘hero’ (titre + badge)
      - la barre de navigation sticky
      - la ‘filterbar’ sticky
      - le style des radio-buttons (tabs)
    Je m’aligne sur une esthétique sombre, sans header Streamlit par défaut.
    """
    blue = UEFA_BLUE
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        :root{{ --bg:#000; --ink:#eaf2ff; --muted:#a7b3d0; --uefa:{blue}; --uefaLight: rgba(31,94,255,.18); }}
        /* Plein écran + fond noir */
        .stApp {{ color: var(--ink); background: var(--bg); }}
        .block-container {{ max-width: 100% !important; padding: 0 18px 24px 18px; }}
        header[data-testid="stHeader"], #MainMenu, footer {{ display:none; }}

        .hero{{
          position: sticky; top: 0; z-index: 999;
          margin: 0 -18px 14px -18px; padding: 22px 26px 18px 26px;
          background: linear-gradient(135deg, rgba(20,20,30,.85), rgba(10,10,18,.85));
          border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(10px);
          box-shadow: 0 18px 40px rgba(0,0,0,.55), inset 0 0 0 1px rgba(255,255,255,.04);
        }}
        .hero h1{{ font: 700 2rem/1.2 "Poppins",system-ui; margin: 6px 0 2px 0; letter-spacing:.3px; }}
        .badge{{ display:inline-block; padding: 4px 12px; border-radius:999px;
                background: var(--uefaLight); border:1px solid rgba(255,255,255,.18);
                font-weight:600; letter-spacing:.02em; color:#dfe7ff; }}
        .tagline{{ margin:0; font:600 1.05rem/1.35 "Poppins",system-ui; color:#dfe7ff; }}

        .navwrap{{ position: sticky; top: 84px; z-index: 998; padding: 8px 12px; border-radius: 14px;
                   background: rgba(255,255,255,.03); backdrop-filter: blur(6px);
                   border: 1px solid rgba(255,255,255,.06); margin-bottom: .6rem; }}
        div[data-baseweb="radio"] > div{{ flex-wrap: wrap; gap:10px; }}
        div[data-baseweb="radio"] label svg {{ display:none !important; }}
        div[data-baseweb="radio"] label div:first-child {{ width:0 !important; margin:0 !important; padding:0 !important; }}
        div[data-baseweb="radio"] label div:last-child {{ margin-left: 0 !important; }}
        div[data-baseweb="radio"] label{{
          color: var(--ink); background: rgba(255,255,255,.05);
          border: 1px solid rgba(255,255,255,.14); border-radius: 10px;
          padding: 9px 14px; font-weight:600; font-size:14.5px; transition: all .18s ease;
          letter-spacing:.2px;
        }}
        div[data-baseweb="radio"] label[data-checked="true"]{{
          background: var(--uefa); color: #03112b; border-color: transparent;
          box-shadow: 0 10px 30px rgba(31,94,255,.35);
        }}
        div[data-baseweb="radio"] label:hover{{ transform: translateY(-1px); border-color: var(--uefa); }}

        .element-container:has(.stDataFrame){{
          background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08);
          border-radius: 14px; padding: 8px 8px 2px 8px;
        }}
        .filterbar{{ position: sticky; top: 134px; z-index: 997; margin-bottom: .6rem;
                    background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
                    padding: 10px 14px; border-radius: 12px; backdrop-filter: blur(6px); }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _header_and_nav():
    """
    J’affiche le header (titre + badge) et une navigation horizontale
    pour découper l’expérience en pages : Overview, Scraper, Processing, etc.
    """
    st.markdown(
        """
        <div class="hero">
          <div class="badge">UEFA Champions League 2023/2024</div>
          <h1>UCL 23/24 — Dashboard</h1>
          <p class="tagline">Et si on revivait la saison 2023-2024 ?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = [
        ("Overview", "Aperçu"),
        ("Scraper", "Scraper"),
        ("Processing", "Processing"),
        ("Match Analysis", "Matchs"),
        ("Team Analysis", "Équipes"),
        ("Player Analysis", "Joueurs"),
        ("Paths & Stories", "Histoires"),
    ]
    st.markdown('<div class="navwrap">', unsafe_allow_html=True)
    choice = st.radio(
        "Navigation",
        options=[k for k, _ in pages],
        format_func=lambda k: dict(pages)[k],
        horizontal=True,
        key="nav",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return choice

# ====================================================
# ===============  Filtres globaux UI  ===============
# ====================================================
def _init_filters(proc):
    """
    Je construis une barre de filtres globale (stage + équipes) qui agit
    sur les différentes pages d’analyse. Les filtres sont stockés
    dans st.session_state pour être persistants pendant la session.
    """
    M = proc.get("matches", pd.DataFrame())
    st.markdown('<div class="filterbar">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2,2,2,1])

    stages = ["All"] + (sorted(M["stage"].dropna().unique().tolist()) if "stage" in M.columns else [])
    teams_list = sorted(pd.unique(pd.concat([
        M.get("home_team", pd.Series(dtype=str)),
        M.get("away_team", pd.Series(dtype=str))
    ], ignore_index=True).dropna()).tolist())

    if "filters" not in st.session_state:
        st.session_state.filters = dict(stage="All", teams=[])

    with c1:
        st.session_state.filters["stage"] = st.selectbox("Filtrer par phase", stages, index=0)
    with c2:
        st.session_state.filters["teams"] = st.multiselect("Filtrer par équipe(s)", teams_list, default=[])
    with c3:
        show_only_knockouts = st.checkbox("Seulement KO", value=False, help="Round of 16 et +")
    with c4:
        if st.button("Réinitialiser"):
            st.session_state.filters = dict(stage="All", teams=[])
    st.markdown('</div>', unsafe_allow_html=True)
    return show_only_knockouts

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Application des filtres sélectionnés sur le DataFrame cible.
    - Stage exact si précisé (sinon 'All' => no-op)
    - Équipe(s) : filtre sur home_team/away_team (ou 'team' si présent)
    """
    if df is None or df.empty:
        return df
    f = st.session_state.get("filters", {})
    out = df.copy()
    if f.get("stage") and f["stage"] != "All" and "stage" in out.columns:
        out = out[out["stage"] == f["stage"]]
    if f.get("teams"):
        t = f["teams"]
        if "home_team" in out.columns and "away_team" in out.columns:
            out = out[(out["home_team"].isin(t)) | (out["away_team"].isin(t))]
        elif "team" in out.columns:
            out = out[out["team"].isin(t)]
    return out

# ====================================================
# ================== Données + UI ====================
# ====================================================
_inject_css_dark()  # J’active le thème sombre custom
viz = UCLVisualizer()  # Je prépare mon module de visualisations (centralise la charte)
data = load_all()      # Je charge tout (raw, processed, metrics)
raw, proc, met = data["raw"], data["proc"], data["metrics"]
section = _header_and_nav()  # Je laisse l’utilisateur choisir la page

# ------------------------ Scraper -------------------
if section == "Scraper":
    st.subheader("Scraper — données UEFA 2023/2024")
    c1, c2 = st.columns(2)
    with c1:
        # Je laisse ajuster la pagination côté API (utile si throttling)
        limit = st.slider("Page size (API limit)", 20, 100, 50, step=10)
        if st.button("Récupérer tous les matchs"):
            with st.spinner("Fetching matches..."):
                dfm = scrape_matches(limit_per_page=limit, verbose=True)
                st.success(f"{len(dfm)} lignes")
                _df(_tidy_for_display(dfm.tail(15), ("group","country")))
    with c2:
        if st.button("Récupérer les classements (groups)"):
            with st.spinner("Fetching groups..."):
                dfg = scrape_groups(verbose=True)
                st.success(f"{len(dfg)} lignes")
                _df(_tidy_for_display(dfg.head(20), ("team_id",)))

# ----------------------- Processing -----------------
elif section == "Processing":
    st.subheader("Nettoyage & Structuration")
    dp = DataProcessor()
    if st.button("Lancer le pipeline (processed + metrics)"):
        with st.spinner("Processing..."):
            dp.compute_and_export_all()  # Ici je calcule et j’exporte toutes les tables processed + metrics
        st.success("OK ✅")

    # Aperçu des tables processed (pour validation visuelle rapide)
    st.markdown("**Aperçu processed**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Matches (processed)")
        _df(_tidy_for_display(proc["matches"].tail(20), ("group","country")))
    with c2:
        st.caption("Standings (processed)")
        _df(_tidy_for_display(proc["standings"].head(32), ("team_id",)))

    # Boutons d’export direct (qualité de vie pour le correcteur / la démo)
    st.markdown("#### Export")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        if not proc["matches"].empty:
            st.download_button("⬇️ processed_matches.csv", proc["matches"].to_csv(index=False).encode("utf-8"),
                               file_name=PROCESSED_MATCHES, mime="text/csv")
    with cc2:
        if not proc["standings"].empty:
            st.download_button("⬇️ processed_standings.csv", proc["standings"].to_csv(index=False).encode("utf-8"),
                               file_name=PROCESSED_STANDINGS, mime="text/csv")
    with cc3:
        if not met["teams"].empty:
            st.download_button("⬇️ team_metrics.csv", met["teams"].to_csv(index=False).encode("utf-8"),
                               file_name=TEAM_METRICS, mime="text/csv")

# ------------------------ Overview ------------------
elif section == "Overview":
    st.subheader("Overview — saison 2023/2024")
    M, S, P, TM = proc["matches"], proc["standings"], proc["players"], met["teams"]
    if M.empty:
        st.warning("Aucun match chargé. Lance d’abord le scraping & processing.")
    else:
        # Je place la filterbar globale (stage/équipes) + un toggle “KO only”
        show_only_ko = _init_filters(proc)
        Mf = _apply_filters(M)
        if show_only_ko and "stage" in Mf.columns:
            Mf = Mf[Mf["stage"].str.contains("Round of 16|Quarter|Semi|Final", case=False, na=False)]

        # KPIs synthétiques (matches, buts, affluence, nb équipes)
        g1, g2, g3, g4 = st.columns(4)
        total_matches = len(Mf)
        total_goals = (pd.to_numeric(Mf.get("home_goals", 0), errors="coerce").fillna(0)
                       + pd.to_numeric(Mf.get("away_goals", 0), errors="coerce").fillna(0)).sum()
        avg = round(total_goals / total_matches, 2) if total_matches else 0
        attendance = pd.to_numeric(Mf.get("attendance", 0), errors="coerce").fillna(0).sum()
        with g1: st.metric("Matches", f"{total_matches}")
        with g2: st.metric("Goals", f"{int(total_goals)}", f"avg {avg}/match")
        with g3: st.metric("Total attendance", f"{attendance:,}")
        with g4:
            st.metric("Teams", f"{proc['standings']['team'].nunique()}" if not proc["standings"].empty else "—")

        # Chronologie + distribution des buts
        c1, c2 = st.columns(2)
        with c1: _p(viz.create_match_timeline(Mf))
        with c2: _p(viz.create_goals_distribution(Mf))

        # Over/Under si dispo
        if hasattr(viz, "create_over_under_chart"):
            st.markdown("### Over / Under 2.5")
            _p(viz.create_over_under_chart(Mf, line=2.5))

        # Évolution par phase (mon visuel ad hoc)
        st.markdown("### Evolution par phase")
        _p(fig_phase_evolution(Mf))

        # Affluence + arbitres (top)
        st.markdown("### Affluences & Arbitres")
        c1, c2 = st.columns(2)
        with c1: _p(viz.create_attendance_top_stadiums(Mf, top_n=10))
        with c2: _p(viz.create_referees_appearances(Mf, top_n=10))

# --------------------- Match Analysis ----------------
elif section == "Match Analysis":
    st.subheader("Match Analysis")
    M = proc["matches"]
    if M.empty:
        st.warning("Matches manquants.")
    else:
        _init_filters(proc)
        df = _apply_filters(M)

        # Ventilation résultats + boxplots buts home/away
        c1, c2 = st.columns(2)
        with c1: _p(viz.create_result_breakdown(df))
        with c2: _p(viz.create_home_away_goals_box(df))

        # Heatmap résultats moyens par adversaire
        st.markdown("### Résultats moyens par adversaire (heatmap)")
        _p(viz.create_results_heatmap(df, min_matches=2))

# --------------------- Team Analysis -----------------
elif section == "Team Analysis":
    st.subheader("Team Analysis")
    M, TM, S = proc["matches"], met["teams"], proc["standings"]
    _init_filters(proc)
    Mf = _apply_filters(M)

    # Comparaisons d’équipes (scatter/style) — fallback sur matches si metrics absentes
    base = TM if not TM.empty else Mf
    c1, c2 = st.columns(2)
    with c1: _p(viz.create_team_comparison(base))
    with c2: _p(viz.create_team_style_scatter(base))

    # Radar des meilleures attaques (paramétrable)
    top_n = st.slider("Top équipes à projeter sur le radar (par GOALS FOR)", 3, 6, 3)
    _p(viz.create_team_radar_chart(base, top_n=top_n))

    # Classements de groupes (tableau riche)
    st.markdown("### Classements de groupes (finals)")
    _p(viz.create_group_standings_table(S))

    # Analyses spécifiques de groupes (points/GD + W/D/L stacked)
    st.markdown("### Analyse des groupes")
    c1, c2 = st.columns(2)
    with c1: _p(viz.create_group_points_gd_scatter(S))
    with c2: _p(viz.create_group_wdl_stacked(S, top_per_group=4))

    # Difficulté de parcours en KO (mon visuel ad hoc)
    st.markdown("### Parcours le plus difficile (KO)")
    _p(fig_path_difficulty(M, TM))

# -------------------- Player Analysis ----------------
elif section == "Player Analysis":
    st.subheader("Player Analysis")
    P, PM = proc["players"], met["players"]
    # Je préfère utiliser la table de metrics joueurs si dispo (plus stable pour les comparaisons)
    df = PM if not PM.empty else P
    if df.empty:
        st.warning("Aucune donnée joueur traitée.")
    else:
        # Tops buteurs / passeurs
        c1, c2 = st.columns(2)
        with c1: _p(viz.create_top_scorers_bar(df, top_n=15))
        with c2: _p(viz.create_top_assists_bar(df, top_n=15))

        # Involvement (G + A) si dispo dans viz.py
        if hasattr(viz, "create_goal_involvement"):
            st.markdown("### Goal Involvement (G + A)")
            _p(viz.create_goal_involvement(df, top_n=15))

        # Relation minutes-buts (efficacité)
        st.markdown("### Minutes vs Goals")
        _p(viz.create_minutes_vs_goals(df))

        # Radar individuel (sélection par nom)
        st.markdown("### Radar d’un joueur")
        default_players = sorted(df["player_name"].dropna().unique().tolist()) if "player_name" in df.columns else []
        player_name = st.selectbox("Choisis un joueur (ou tape)", options=[""] + default_players, index=0,
                                   placeholder="Kylian Mbappé, Jude Bellingham…")
        if player_name:
            _p(viz.create_player_radar(df, player_name))

# ------------------- Paths & Stories -----------------
elif section == "Paths & Stories":
    st.subheader("Paths & Stories")
    M, TM = proc["matches"], met["teams"]
    if M.empty:
        st.warning("Matches manquants.")
    else:
        _init_filters(proc)
        Mf = _apply_filters(M)

        # Parcours du champion (ex : Real Madrid) — storytelling
        st.markdown("### Parcours du Real Madrid jusqu’au titre")
        _p(viz.create_team_progression(Mf, "Real Madrid"))

        # Momentum (forme glissante)
        st.markdown("### Momentum d’équipe (rolling)")
        team_for_momentum = st.selectbox("Équipe", sorted(pd.unique(pd.concat([Mf["home_team"], Mf["away_team"]]).dropna()).tolist()))
        _p(viz.create_team_momentum(Mf, team_for_momentum, window=3))

        # Surprises/déceptions (mon visuel ad hoc basé sur ranks)
        st.markdown("### Surprises et déceptions de la saison")
        _p(fig_upsets(M, TM, top_n=12))

        # Sunburst (répartition des buts par phase → équipe)
        st.markdown("### Répartition des buts par phase → équipe (Sunburst)")
        _p(viz.create_phase_sunburst(M))

        # Bracket (flow) — reconstruit depuis les scores cumulés
        st.markdown("### Bracket (flow)")
        bracket = build_bracket_data(M)
        _p(viz.create_tournament_bracket(bracket))
