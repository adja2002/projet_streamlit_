"""
DataProcessor — pipeline 'processing' & 'metrics' UCL
====================================================
ici J’uniformise, nettoie et enrichis les CSV bruts pour produire :
- processed_matches.csv
- processed_standings.csv
- processed_players.csv
- processed_teams.csv
- metrics/*.csv (teams, players, matches)

Philosophie :
- tolérant aux colonnes manquantes / noms variables,
- conversions robustes (int, dates, strings),
- suppression des colonnes 100% vides,
- reconstruction des standings depuis les matchs en fallback,
- calcul de métriques utiles pour les viz.
"""

from __future__ import annotations

import re, math, logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# --- Chemins & noms de fichiers depuis la config (centralisation)
from config import (
    csv_path_raw, csv_path_processed, csv_path_metrics,
    MATCHES_FILE, STANDINGS_FILE, PLAYERS_FILE, TEAMS_FILE,
    PROCESSED_MATCHES, PROCESSED_STANDINGS, PROCESSED_PLAYERS, PROCESSED_TEAMS,
    TEAM_METRICS, PLAYER_METRICS, MATCH_METRICS,
)

# --- Optionnels : je définis des valeurs par défaut si absents de la config
try:
    from config import PLAYER_RANKING_FILE
except Exception:
    PLAYER_RANKING_FILE = "data/raw/player_ranking_2024.csv"

try:
    from config import PROCESSED_PLAYER_RANKING
except Exception:
    PROCESSED_PLAYER_RANKING = "processed_player_ranking.csv"

try:
    from config import MERGED_PLAYERS_FILE
except Exception:
    MERGED_PLAYERS_FILE = "data/processed/players_match_plus_ranking.csv"

# STAGE_MAP optionnel pour normaliser les phases
try:
    from config import STAGE_MAP
except Exception:
    STAGE_MAP = {}

# --- Logger simple et lisible pour le notebook/CLI
logger = logging.getLogger("ucl.dataproc")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class DataProcessor:
    """
    Je regroupe ici toutes les étapes de traitement et export.
    L’init s’assure que les dossiers 'processed' et 'metrics' existent.
    """
    def __init__(self):
        Path(csv_path_processed(PROCESSED_MATCHES)).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_path_metrics(TEAM_METRICS)).parent.mkdir(parents=True, exist_ok=True)

    # ---------------- utils (nettoyage & conversions) ----------------
    @staticmethod
    def _ensure_cols(df: pd.DataFrame, cols: Dict[str, object]) -> pd.DataFrame:
        """Garantit la présence de colonnes avec des valeurs par défaut si manquantes."""
        for c, default in cols.items():
            if c not in df.columns:
                df[c] = default
        return df

    @staticmethod
    def _clean_str(x) -> Optional[str]:
        """Nettoie une chaîne (trim, None pour valeurs vides/placeholder)."""
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "nan", "null", "n/a"}:
            return None
        return s

    @staticmethod
    def _title_or_same(s: Optional[str]) -> Optional[str]:
        """Met en Title Case sauf pour sigles courts (ex: PSG)."""
        if not s:
            return s
        if s.isupper() and len(s) <= 4:  # H / A / PSG…
            return s
        return s.title()

    @staticmethod
    def _to_int(x) -> Optional[int]:
        """
        Convertit vers int de manière robuste :
        - gère '2-0', '2 – 0', '2:0'
        - gère '2.0' / espaces / virgules
        """
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        try:
            s = str(x).strip()
            # cas "2-0", "2 – 0", "2:0"
            for sep in ["–", "-", "—", ":"]:
                if sep in s:
                    parts = s.replace(" ", "").split(sep)
                    if len(parts) >= 2 and parts[0].isdigit():
                        return int(parts[0])
            s = s.replace(" ", "").replace(",", "")
            if s == "":
                return None
            if "." in s and s.count(".") == 1:
                s = s.split(".", 1)[0]
            return int(s)
        except Exception:
            return None

    @staticmethod
    def _to_int_or_nan(x):
        """Version int → NaN autorisé (pour colonnes 'Int64')."""
        v = DataProcessor._to_int(x)
        return np.int64(v) if v is not None else np.nan

    @staticmethod
    def _parse_dt(x):
        """Parse date ISO/UEFA → pandas datetime (UTC)."""
        s = DataProcessor._clean_str(x)
        if not s:
            return pd.NaT
        return pd.to_datetime(str(s).replace("Z", "+00:00"), utc=True, errors="coerce")

    @staticmethod
    def _col_all_blank(series: pd.Series) -> bool:
        """True si la colonne est entièrement vide/NA/placeholder."""
        if series is None:
            return True
        s = series.copy()
        mask = s.isna() | s.astype(str).str.strip().isin(["", "None", "none", "NaN", "nan", "Null", "null"])
        return bool(mask.all())

    @staticmethod
    def _drop_blank_columns(df: pd.DataFrame, *, keep: Optional[set] = None,
                            min_non_null_ratio: float = 0.0) -> pd.DataFrame:
        """
        Supprime les colonnes :
          - 100 % vides (NA/vides/None), et
          - option : quasi vides (non_null_ratio < seuil).
        'keep' = colonnes protégées.
        """
        if df is None or df.empty:
            return df
        keep = keep or set()
        n = len(df)
        to_drop = []
        for c in df.columns:
            if c in keep:
                continue
            s = df[c]
            if DataProcessor._col_all_blank(s):
                to_drop.append(c)
                continue
            if min_non_null_ratio > 0:
                if float(s.notna().sum()) / max(1, n) < min_non_null_ratio:
                    to_drop.append(c)
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"[OK] colonnes supprimées: {to_drop}")
        return df

    # ---------------- IO helpers (lectures tolérantes) ----------------
    def _read_csv_safe(self, path: Path | str) -> Optional[pd.DataFrame]:
        """Lecture CSV silencieuse (retourne None si absent/vide)."""
        try:
            p = Path(path)
            if p.exists() and p.stat().st_size > 0:
                return pd.read_csv(p, encoding="utf-8")
        except Exception as e:
            logger.warning(f"[WARN] lecture CSV échouée {path}: {e}")
        return None

    # Charge les différentes tables (raw → processed fallback)
    def load_matches(self) -> Optional[pd.DataFrame]:
        df = self._read_csv_safe(csv_path_processed(PROCESSED_MATCHES))
        if df is None:
            df = self._read_csv_safe(csv_path_raw(MATCHES_FILE))
        return df

    def load_players(self) -> Optional[pd.DataFrame]:
        df = self._read_csv_safe(csv_path_processed(PROCESSED_PLAYERS))
        if df is None:
            df = self._read_csv_safe(csv_path_raw(PLAYERS_FILE))
        return df

    def load_player_ranking(self) -> Optional[pd.DataFrame]:
        df = self._read_csv_safe(csv_path_processed(PROCESSED_PLAYER_RANKING))
        if df is None:
            df = self._read_csv_safe(PLAYER_RANKING_FILE)
        return df

    def load_players_matchrows(self) -> Optional[pd.DataFrame]:
        return self._read_csv_safe(csv_path_raw(PLAYERS_FILE))

    def load_players_merged(self) -> Optional[pd.DataFrame]:
        return self._read_csv_safe(MERGED_PLAYERS_FILE)

    def load_teams(self) -> Optional[pd.DataFrame]:
        df = self._read_csv_safe(csv_path_processed(PROCESSED_TEAMS))
        if df is None:
            df = self._read_csv_safe(csv_path_raw(TEAMS_FILE))
        return df

    def load_standings(self) -> Optional[pd.DataFrame]:
        df = self._read_csv_safe(csv_path_processed(PROCESSED_STANDINGS))
        if df is None:
            df = self._read_csv_safe(csv_path_raw(STANDINGS_FILE))
        return df

    # ---------------- Reconstruction des standings (fallback) ----------------
    def _rebuild_standings_from_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Si standings brut absent, je reconstruis à partir des matchs de groupes.
        Approche :
          - normalisation colonnes (home/away, scores, group)
          - filtrage sur Group A-H
          - agrégats W/D/L/GF/GA/PTS
        """
        if matches_df is None or matches_df.empty:
            logger.warning("[WARN] Impossible de reconstruire les standings: matches_df vide")
            return pd.DataFrame()

        df = matches_df.copy()

        # Harmonisation possible des noms de colonnes (variantes)
        rename_map = {
            "homeTeam": "home_team",
            "awayTeam": "away_team",
            "home_score": "home_goals",
            "away_score": "away_goals",
            "home": "home_team",
            "away": "away_team",
            "groupName": "group",
            "grp": "group",
            "pool": "group",
        }
        for a, b in rename_map.items():
            if a in df.columns and b not in df.columns:
                df.rename(columns={a: b}, inplace=True)

        # Si 'score' = "2-0", je split en home_goals/away_goals
        if "score" in df.columns and ("home_goals" not in df.columns or "away_goals" not in df.columns):
            hg, ag = [], []
            for val in df["score"].astype(str):
                parts = [p for p in re.split(r"[–—\-:]", val) if p.strip() != ""]
                if len(parts) >= 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    hg.append(int(parts[0])); ag.append(int(parts[1]))
                else:
                    hg.append(np.nan); ag.append(np.nan)
            if "home_goals" not in df.columns: df["home_goals"] = hg
            if "away_goals" not in df.columns: df["away_goals"] = ag

        if "group" not in df.columns:
            df["group"] = "Group Stage"

        # Vérifs indispensables
        for c in ("home_goals", "away_goals"):
            if c not in df.columns:
                logger.warning(f"[WARN] Colonne manquante: {c}")
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in ["home_team", "away_team"]:
            if c not in df.columns:
                logger.error(f"[ERR] Colonne manquante: {c}")
                return pd.DataFrame()
            df[c] = df[c].astype(str)

        usable = df.dropna(subset=["home_goals", "away_goals"])
        if usable.empty:
            logger.error("[ERR] Aucun match avec score exploitable")
            return pd.DataFrame()

        # Je garde uniquement Group A-H s’ils existent (évite KO)
        mask_groups = usable["group"].astype(str).str.contains(r"Group\s+[A-H]", case=False, regex=True)
        if mask_groups.any():
            usable = usable[mask_groups]

        # Deux lignes par match (une par équipe)
        rows = []
        for _, r in usable.iterrows():
            grp = r["group"]
            ht, at = r["home_team"], r["away_team"]
            hg, ag = int(r["home_goals"]), int(r["away_goals"])
            rows.append(dict(group=grp, team=ht, wins=int(hg > ag), draws=int(hg == ag), losses=int(hg < ag),
                             goals_for=hg, goals_against=ag, points=(3 if hg > ag else (1 if hg == ag else 0))))
            rows.append(dict(group=grp, team=at, wins=int(ag > hg), draws=int(ag == hg), losses=int(ag < hg),
                             goals_for=ag, goals_against=hg, points=(3 if ag > hg else (1 if ag == hg else 0))))

        tab = pd.DataFrame(rows)
        if tab.empty:
            logger.error("[ERR] Reconstruction standings: vide")
            return pd.DataFrame()

        # Agrégats finaux + ordonnancement
        tab = tab.groupby(["group", "team"], as_index=False).sum(numeric_only=True)
        tab["played"] = tab["wins"] + tab["draws"] + tab["losses"]
        tab["goal_difference"] = tab["goals_for"] - tab["goals_against"]
        tab = tab.sort_values(["group", "points", "goal_difference", "goals_for"],
                              ascending=[True, False, False, False])
        tab["position"] = tab.groupby("group").cumcount() + 1

        tab = tab[["group","position","team","played","wins","draws","losses",
                   "goals_for","goals_against","goal_difference","points"]].reset_index(drop=True)
        return tab

    # ---------------- Traitements principaux ----------------
    def process_matches(self) -> pd.DataFrame:
        """
        Nettoie et normalise matches.csv :
          - renomme colonnes selon nos conventions,
          - convertit dates/scores,
          - calcule features (result, total_goals, goal_diff, flags home/draw/away),
          - génère *_diff si *_home/*_away existent,
          - supprime colonnes vides,
          - exporte processed_matches.csv.
        """
        raw_path = csv_path_raw(MATCHES_FILE)
        out_path = csv_path_processed(PROCESSED_MATCHES)
        df = pd.read_csv(raw_path, encoding="utf-8")

        # Harmonisation des colonnes (si données viennent de sources légèrement différentes)
        rename_map = {
            "date": "utc_date",
            "team_home": "home_team",
            "team_away": "away_team",
            "score_home": "home_score",
            "score_away": "away_score",
            "phase": "stage",
            "Group": "group",
            "Country": "country",
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        # Colonnes minimales attendues + placeholders
        df = self._ensure_cols(df, {
            "match_id": None, "season": None, "utc_date": None, "stage": None, "group": None,
            "home_team": None, "away_team": None, "home_score": None, "away_score": None,
            "stadium": None, "city": None, "country": None, "referee": None, "attendance": None
        })

        # Normalisation des phases si STAGE_MAP fourni
        if "stage" in df.columns:
            df["stage"] = df["stage"].map(lambda s: STAGE_MAP.get(str(s), s))

        # Nettoyage des strings + TitleCase sur entités
        for col in ["season", "stage", "group", "home_team", "away_team", "stadium", "city", "country", "referee"]:
            df[col] = df[col].map(self._clean_str)
        for col in ["home_team", "away_team", "stadium", "city", "country", "referee"]:
            df[col] = df[col].map(self._title_or_same)

        # Numériques robustes
        for col in ["home_score", "away_score", "attendance"]:
            df[col] = df[col].map(self._to_int).astype("Int64")

        # Dates et features temporelles (UTC)
        df["date_utc"] = df["utc_date"].map(self._parse_dt)
        df["date"] = df["date_utc"]
        df["year"] = df["date_utc"].dt.year
        df["month"] = df["date_utc"].dt.month
        df["weekday"] = df["date_utc"].dt.weekday

        # Résultat 'H/A/D'
        def _result(row):
            hs, as_ = row["home_score"], row["away_score"]
            if pd.isna(hs) or pd.isna(as_): return None
            if hs > as_: return "H"
            if hs < as_: return "A"
            return "D"

        df["result"] = df.apply(_result, axis=1)
        df["home_goals"] = df["home_score"].astype("Int64")
        df["away_goals"] = df["away_score"].astype("Int64")
        df["total_goals"] = (df["home_goals"].fillna(0) + df["away_goals"].fillna(0)).astype("Int64")
        df["goal_diff"] = (df["home_goals"].fillna(0) - df["away_goals"].fillna(0)).astype("int64")
        df["home_win"] = (df["result"] == "H").astype("Int64")
        df["draw"] = (df["result"] == "D").astype("Int64")
        df["away_win"] = (df["result"] == "A").astype("Int64")

        # *_diff auto si l’on dispose de *_home/*_away (ex: xg_home, xg_away)
        stat_pairs = [c[:-5] for c in df.columns if c.endswith("_home")]
        for base in stat_pairs:
            h = f"{base}_home"; a = f"{base}_away"; d = f"{base}_diff"
            try:
                hh = pd.to_numeric(df[h], errors="coerce")
                aa = pd.to_numeric(df[a], errors="coerce")
                df[d] = (hh - aa).round(3)
            except Exception:
                pass

        # Suppression colonnes 100% vides (en protégeant l’essentiel)
        df = self._drop_blank_columns(
            df,
            keep={"match_id","utc_date","date_utc","date","stage","group",
                  "home_team","away_team","home_score","away_score",
                  "home_goals","away_goals","total_goals","result"},
            min_non_null_ratio=0.0
        )

        # Dé-duplication et tri temporel
        df = df.drop_duplicates(subset=["match_id"]).sort_values(["date_utc", "match_id"]).reset_index(drop=True)

        # Export
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"[OK] processed matches -> {out_path} ({len(df)} rows)")
        return df

    def process_standings(self) -> pd.DataFrame:
        """
        Traite standings.csv :
          - reconstruit depuis matches si manquant,
          - convertit les numériques,
          - vérifie goal_difference,
          - ordonne par groupe puis recalcule position,
          - supprime colonnes vides,
          - exporte processed_standings.csv.
        """
        raw_path = csv_path_raw(STANDINGS_FILE)
        out_path = csv_path_processed(PROCESSED_STANDINGS)

        raw_df = self._read_csv_safe(raw_path)

        # Fallback : reconstruction depuis les matchs si standings vide
        if raw_df is None or raw_df.empty:
            logger.warning("[WARN] standings bruts manquants/vides → reconstruction depuis matches.csv")
            matches_df = self.load_matches()
            if matches_df is None:
                matches_df = self._read_csv_safe(csv_path_raw(MATCHES_FILE))
            rebuilt = self._rebuild_standings_from_matches(matches_df if matches_df is not None else pd.DataFrame())
            if rebuilt is None or rebuilt.empty:
                logger.error("[ERR] Reconstruction standings impossible (matches insuffisants)")
                return pd.DataFrame()
            rebuilt.to_csv(raw_path, index=False, encoding="utf-8")
            raw_df = rebuilt
            logger.info(f"[OK] standings reconstruits -> {raw_path} ({len(raw_df)} lignes)")

        # Normalisation & conversions
        df = raw_df.copy()
        df = self._ensure_cols(df, {
            "group": None, "position": None, "team_id": None, "team": None,
            "played": None, "wins": None, "draws": None, "losses": None,
            "goals_for": None, "goals_against": None, "goal_difference": None,
            "points": None
        })

        for col in ["group", "team"]:
            df[col] = df[col].map(self._clean_str)

        nums = ["position","team_id","played","wins","draws","losses","goals_for","goals_against","goal_difference","points"]
        for c in nums:
            df[c] = df[c].map(self._to_int_or_nan).astype("Int64")

        # Recalcule GD si manquant et si possible
        if "goal_difference" in df.columns:
            need_fix = df["goal_difference"].isna()
            if "goals_for" in df.columns and "goals_against" in df.columns and need_fix.any():
                df.loc[need_fix, "goal_difference"] = (
                    df.loc[need_fix, "goals_for"].astype("float") - df.loc[need_fix, "goals_against"].astype("float")
                ).astype("Int64")

        # Si team_id est vide partout, on l’enlève
        use_team_id = "team_id" in df.columns and not self._col_all_blank(df["team_id"])
        if not use_team_id and "team_id" in df.columns:
            df = df.drop(columns=["team_id"])
            logger.info("[OK] colonne vide supprimée dans processed_standings: team_id")

        # Ordonnancement : points > goal_diff > goals_for ; position recalculée par groupe
        subset = ["group","team_id"] if use_team_id else ["group","team"]
        df = df.drop_duplicates(subset=subset).sort_values(
            ["group", "points", "goal_difference", "goals_for"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

        df["position"] = df.groupby("group").cumcount() + 1

        # Suppression colonnes vides restantes
        df = self._drop_blank_columns(df, keep=set(subset) | {"position","played","wins","draws","losses","points"}, min_non_null_ratio=0.0)

        # Export
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"[OK] processed standings -> {out_path} ({len(df)} rows)")
        return df

    # ---------------- Nouveau: processing du ranking joueurs (compstats) ----------------
    def process_player_ranking(self) -> pd.DataFrame:
        """
        Traite un 'player_ranking' brut (compstats) :
          - harmonise noms de colonnes,
          - conversions num,
          - dérive per90 + impact_index,
          - drop colonnes vides,
          - exporte processed_player_ranking.csv.
        """
        raw_df = self._read_csv_safe(PLAYER_RANKING_FILE)
        if raw_df is None or raw_df.empty:
            logger.warning("[WARN] player_ranking brut absent → skip")
            return pd.DataFrame()

        df = raw_df.copy()
        rename_map = {
            "player": "player_name",
            "minutes_played": "minutes",
            "yellow_cards": "yellow",
            "red_cards": "red",
            "team": "team_name",
        }
        for a, b in rename_map.items():
            if a in df.columns and b not in df.columns:
                df.rename(columns={a: b}, inplace=True)

        base_cols = ["player_id","player_name","team_name","goals","assists","minutes","yellow","red"]
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan

        for c in ["goals","assists","minutes","yellow","red"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        def per90(num, minutes):
            m = np.where(minutes > 0, minutes, np.nan)
            return (num * 90 / m).round(3)

        if "goals" in df.columns and "minutes" in df.columns:
            df["goals_per90"] = per90(df["goals"], df["minutes"])
        if "assists" in df.columns and "minutes" in df.columns:
            df["assists_per90"] = per90(df["assists"], df["minutes"])
        has_cols = [c for c in ["goals_per90","assists_per90"] if c in df.columns]
        if has_cols:
            df["impact_index"] = df[has_cols].sum(axis=1).round(3)

        out = csv_path_processed(PROCESSED_PLAYER_RANKING)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        df = self._drop_blank_columns(df, keep={"player_name","team_name","goals","assists","minutes","yellow","red","goals_per90","assists_per90","impact_index"})
        df.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"[OK] processed player ranking -> {out} ({len(df)} rows)")
        return df

    def process_players(self) -> pd.DataFrame:
        """
        Unifie la table joueurs traitée. Priorité :
          1) table fusionnée match+ranking (MERGED_PLAYERS_FILE)
          2) stats par match (PLAYERS_FILE)
          3) ranking (PLAYER_RANKING_FILE)
        Dans tous les cas :
          - harmonise colonnes,
          - dérive per90 + impact_index,
          - 'team_name' → 'team' pour la cohérence viz,
          - drop colonnes vides,
          - exporte processed_players.csv.
        """
        out_path = csv_path_processed(PROCESSED_PLAYERS)

        # 1) Fusion prête
        merged = self.load_players_merged()
        if merged is not None and not merged.empty:
            df = merged.copy()
            rename_map = {
                "team": "team_name",
                "player": "player_name",
                "minutes_played": "minutes",
                "yellow_cards": "yellow",
                "red_cards": "red",
            }
            for a, b in rename_map.items():
                if a in df.columns and b not in df.columns:
                    df.rename(columns={a: b}, inplace=True)

            for c in ["goals","assists","minutes","yellow","red"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            def per90(num, minutes):
                m = np.where(minutes > 0, minutes, np.nan)
                return (num * 90 / m).round(3)

            if "goals" in df.columns and "minutes" in df.columns:
                df["goals_per90"] = per90(df["goals"], df["minutes"])
            if "assists" in df.columns and "minutes" in df.columns:
                df["assists_per90"] = per90(df["assists"], df["minutes"])
            has_cols = [c for c in ["goals_per90","assists_per90"] if c in df.columns]
            if has_cols:
                df["impact_index"] = df[has_cols].sum(axis=1).round(3)

            # Harmoniser team_name -> team (utilisé par les viz)
            if "team_name" in df.columns and "team" not in df.columns:
                df = df.rename(columns={"team_name": "team"})

            df = self._drop_blank_columns(
                df,
                keep={"category","player_name","team","position","goals","assists","minutes","yellow","red","matches","goals_per90","assists_per90","impact_index"},
                min_non_null_ratio=0.0
            )

            df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"[OK] processed players (from merged) -> {out_path} ({len(df)} rows)")
            return df

        # 2) Stats par match
        pmatch = self.load_players_matchrows()
        if pmatch is not None and not pmatch.empty:
            df = pmatch.copy()
            rename_map = {"player": "player_name", "team": "team_name",
                          "minutes_played": "minutes", "yellow_cards": "yellow", "red_cards": "red"}
            for a, b in rename_map.items():
                if a in df.columns and b not in df.columns:
                    df.rename(columns={a: b}, inplace=True)

            for c in ["minutes","goals","assists","yellow","red"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            def per90(num, minutes):
                m = np.where(minutes > 0, minutes, np.nan)
                return (num * 90 / m).round(3)

            if "goals" in df.columns and "minutes" in df.columns:
                df["goals_per90"] = per90(df["goals"], df["minutes"])
            if "assists" in df.columns and "minutes" in df.columns:
                df["assists_per90"] = per90(df["assists"], df["minutes"])
            has_cols = [c for c in ["goals_per90","assists_per90"] if c in df.columns]
            if has_cols:
                df["impact_index"] = df[has_cols].sum(axis=1).round(3)

            # team_name -> team
            if "team_name" in df.columns and "team" not in df.columns:
                df = df.rename(columns={"team_name": "team"})

            df = self._drop_blank_columns(
                df,
                keep={"category","player_name","team","position","goals","assists","minutes","yellow","red","matches","goals_per90","assists_per90","impact_index"},
                min_non_null_ratio=0.0
            )

            df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"[OK] processed players (from per-match) -> {out_path} ({len(df)} rows)")
            return df

        # 3) Ranking seul
        prank = self.load_player_ranking()
        if prank is not None and not prank.empty:
            df = prank.copy()
            rename_map = {
                "player": "player_name", "team": "team_name",
                "minutes_played": "minutes", "yellow_cards": "yellow", "red_cards": "red",
            }
            for a, b in rename_map.items():
                if a in df.columns and b not in df.columns:
                    df.rename(columns={a: b}, inplace=True)

            for c in ["goals","assists","minutes","yellow","red"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            def per90(num, minutes):
                m = np.where(minutes > 0, minutes, np.nan)
                return (num * 90 / m).round(3)

            if "goals" in df.columns and "minutes" in df.columns:
                df["goals_per90"] = per90(df["goals"], df["minutes"])
            if "assists" in df.columns and "minutes" in df.columns:
                df["assists_per90"] = per90(df["assists"], df["minutes"])
            has_cols = [c for c in ["goals_per90","assists_per90"] if c in df.columns]
            if has_cols:
                df["impact_index"] = df[has_cols].sum(axis=1).round(3)

            if "team_name" in df.columns and "team" not in df.columns:
                df = df.rename(columns={"team_name": "team"})

            df = self._drop_blank_columns(
                df,
                keep={"category","player_name","team","position","goals","assists","minutes","yellow","red","matches","goals_per90","assists_per90","impact_index"},
                min_non_null_ratio=0.0
            )

            df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"[OK] processed players (from ranking) -> {out_path} ({len(df)} rows)")
            return df

        # Fallback vide si aucune source détectée
        empty = pd.DataFrame(columns=["category","player_name","team","position","goals","assists","minutes","yellow","red"])
        empty.to_csv(out_path, index=False, encoding="utf-8")
        logger.warning("[WARN] aucun fichier joueur trouvé → export d’un tableau vide traité")
        return empty

    def process_teams(self) -> pd.DataFrame:
        """
        Construit processed_teams.csv à partir de standings (prioritaire),
        puis fallback depuis matches. teams.csv brut peut enrichir (id/pays).
        """
        out_path = csv_path_processed(PROCESSED_TEAMS)

        # 1) Depuis standings
        stand = self.load_standings()
        if stand is None or stand.empty:
            stand = self._read_csv_safe(csv_path_raw(STANDINGS_FILE))

        if stand is not None and not stand.empty and "team" in stand.columns:
            s = stand.copy()
            s["team"] = s["team"].map(self._clean_str)
            if "group" not in s.columns:
                s["group"] = None

            # 'journey' = groupe majoritaire (utile pour les viz 'parcours')
            journey_map = (
                s.dropna(subset=["team"])
                 .groupby("team")["group"]
                 .agg(lambda x: x.dropna().mode().iloc[0] if x.dropna().size else None)
                 .to_dict()
            )

            teams_df = (
                pd.DataFrame({"team": sorted(t for t in s["team"].dropna().unique())})
                .assign(team_id=pd.Series(dtype="Int64"))
                .assign(country=None)
                .assign(journey=lambda d: d["team"].map(journey_map))
            )

            # Enrichissement facultatif depuis teams.csv brut
            raw_teams = self._read_csv_safe(csv_path_raw(TEAMS_FILE))
            if raw_teams is not None and not raw_teams.empty:
                rt = raw_teams.copy()
                alias = {}
                if "name" in rt.columns and "team" not in rt.columns: alias["name"] = "team"
                if "id" in rt.columns and "team_id" not in rt.columns: alias["id"] = "team_id"
                if alias:
                    rt = rt.rename(columns=alias)

                keep_cols = [c for c in ["team", "team_id", "country", "journey"] if c in rt.columns]
                if "team" in keep_cols:
                    rt["team"] = rt["team"].map(self._clean_str)
                    teams_df = teams_df.merge(rt[keep_cols], on="team", how="left", suffixes=("", "_raw"))
                    # Merge propre (priorité aux valeurs déjà présentes)
                    if "team_id_raw" in teams_df.columns:
                        teams_df["team_id"] = teams_df["team_id"].fillna(teams_df["team_id_raw"])
                        teams_df = teams_df.drop(columns=["team_id_raw"])
                    if "country_raw" in teams_df.columns:
                        teams_df["country"] = teams_df["country"].where(teams_df["country"].notna(), teams_df["country_raw"])
                        teams_df = teams_df.drop(columns=["country_raw"])
                    if "journey_raw" in teams_df.columns:
                        teams_df["journey"] = teams_df["journey"].where(teams_df["journey"].notna(), teams_df["journey_raw"])
                        teams_df = teams_df.drop(columns=["journey_raw"])

            # Nettoyages finaux
            teams_df["team_id"] = teams_df["team_id"].map(DataProcessor._to_int_or_nan).astype("Int64")
            teams_df["country"] = teams_df["country"].map(self._clean_str)
            teams_df["journey"] = teams_df["journey"].map(self._clean_str)

            teams_df = teams_df.drop_duplicates(subset=["team_id", "team"]).reset_index(drop=True)
            teams_df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"[OK] processed teams (from standings) -> {out_path} ({len(teams_df)} rows)")
            return teams_df

        # 2) Fallback : depuis matches
        matches = self.load_matches()
        if matches is not None and not matches.empty:
            m = matches.copy()
            rename_map = {"homeTeam": "home_team", "awayTeam": "away_team"}
            for a, b in rename_map.items():
                if a in m.columns and b not in m.columns:
                    m.rename(columns={a: b}, inplace=True)

            if "home_team" in m.columns and "away_team" in m.columns:
                teams = pd.unique(pd.concat([m["home_team"], m["away_team"]], ignore_index=True).dropna())
                teams_df = pd.DataFrame({"team": sorted(map(str, teams))})
                teams_df["team_id"] = pd.Series(dtype="Int64")
                teams_df["country"] = None
                teams_df["journey"] = None
                teams_df.to_csv(out_path, index=False, encoding="utf-8")
                logger.info(f"[OK] processed teams (from matches) -> {out_path} ({len(teams_df)} rows)")
                return teams_df

        # 3) Vide si rien
        empty = pd.DataFrame(columns=["team_id","team","country","journey"])
        empty.to_csv(out_path, index=False, encoding="utf-8")
        logger.warning("[WARN] teams introuvables → export d’un fichier traité vide")
        return empty

    # ---------------- Metrics (équipes, joueurs, matchs) ----------------
    @staticmethod
    def _points(hg: int, ag: int, side: str) -> int:
        """Calcule les points (3/1/0) selon côté home/away et score."""
        if side == "home":
            if hg > ag: return 3
            if hg == ag: return 1
            return 0
        else:
            if ag > hg: return 3
            if ag == hg: return 1
            return 0

    def build_team_metrics(self, matches_df: pd.DataFrame, standings_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Agrège les performances par équipe :
          - matches, buts pour/contre, points,
          - goal_diff, moyennes par match, points_per_game,
          - wins/draws/losses scindés home/away puis sommés,
          - merge optionnel avec position finale (standings).
        Export : metrics/team_metrics.csv
        """
        if matches_df is None or matches_df.empty:
            return pd.DataFrame()

        need = {"home_team","away_team","home_goals","away_goals"}
        if not need.issubset(matches_df.columns):
            rename = {}
            if "home_score" in matches_df.columns: rename["home_score"] = "home_goals"
            if "away_score" in matches_df.columns: rename["away_score"] = "away_goals"
            matches_df = matches_df.rename(columns=rename)
        matches = matches_df.copy()

        # Lignes 'home'
        home_rows = matches[["group","stage","home_team","home_goals","away_goals"]].copy()
        home_rows.columns = ["group","stage","team","gf","ga"]
        home_rows["points"] = home_rows.apply(lambda r: self._points(r["gf"], r["ga"], "home"), axis=1)

        # Lignes 'away' (on inverse gf/ga pour la perspective de l'équipe en déplacement)
        away_rows = matches[["group","stage","away_team","home_goals","away_goals"]].copy()
        away_rows.columns = ["group","stage","team","hg","ag"]
        away_rows.rename(columns={"hg":"ga","ag":"gf"}, inplace=True)
        away_rows["points"] = away_rows.apply(lambda r: self._points(r["ga"], r["gf"], "away"), axis=1)

        long_df = pd.concat([home_rows, away_rows], ignore_index=True)

        # Agrégats principaux
        agg = long_df.groupby(["team"], dropna=True).agg(
            matches=("team","size"),
            goals_for=("gf","sum"),
            goals_against=("ga","sum"),
            points=("points","sum"),
        ).reset_index()
        agg["goal_diff"] = agg["goals_for"] - agg["goals_against"]
        agg["avg_goals_for"] = (agg["goals_for"] / agg["matches"]).round(3)
        agg["avg_goals_against"] = (agg["goals_against"] / agg["matches"]).round(3)
        agg["points_per_game"] = (agg["points"] / agg["matches"]).round(3)

        # Wins / Draws / Losses par côté, puis fusion
        def res_cols(df: pd.DataFrame, side: str) -> pd.DataFrame:
            if side == "home":
                w = (df["home_goals"] > df["away_goals"]).astype(int)
                d = (df["home_goals"] == df["away_goals"]).astype(int)
                l = (df["home_goals"] < df["away_goals"]).astype(int)
                t = df["home_team"]
            else:
                w = (df["away_goals"] > df["home_goals"]).astype(int)
                d = (df["away_goals"] == df["home_goals"]).astype(int)
                l = (df["away_goals"] < df["home_goals"]).astype(int)
                t = df["away_team"]
            out = pd.DataFrame({"team": t, "w": w, "d": d, "l": l})
            return out.groupby("team").sum().reset_index()

        res = res_cols(matches, "home").merge(res_cols(matches, "away"), on="team", suffixes=("_h","_a"))
        res["wins"] = res["w_h"] + res["w_a"]
        res["draws"] = res["d_h"] + res["d_a"]
        res["losses"] = res["l_h"] + res["l_a"]
        res = res[["team","wins","draws","losses"]]

        # Position (si standings dispo)
        if standings_df is not None and not standings_df.empty:
            keep = standings_df[["group","team","position"]].dropna()
            agg = agg.merge(keep, on="team", how="left")

        # Fusion finale + tri
        agg = agg.merge(res, on="team", how="left")
        agg = agg.sort_values(["points","goal_diff","goals_for"], ascending=[False, False, False]).reset_index(drop=True)

        out = csv_path_metrics(TEAM_METRICS)
        agg.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"[OK] team metrics -> {out} ({len(agg)} rows)")
        return agg

    def build_player_metrics(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les métriques joueurs (per90 + impact_index) à partir de processed_players.
        Export : metrics/player_metrics.csv
        """
        if players_df is None or players_df.empty:
            return pd.DataFrame()
        df = players_df.copy()

        # Harmonisation
        rename = {}
        if "minutes_played" in df.columns and "minutes" not in df.columns:
            rename["minutes_played"] = "minutes"
        if "player" in df.columns and "player_name" not in df.columns:
            rename["player"] = "player_name"
        if "team" in df.columns and "team_name" not in df.columns:
            rename["team"] = "team_name"
        if rename:
            df = df.rename(columns=rename)

        for col in ["goals","assists","minutes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        def per90(num, minutes):
            m = np.where(minutes > 0, minutes, np.nan)
            return (num * 90 / m).round(3)

        if "goals" in df.columns and "minutes" in df.columns:
            df["goals_per90"] = per90(df["goals"], df["minutes"])
        if "assists" in df.columns and "minutes" in df.columns:
            df["assists_per90"] = per90(df["assists"], df["minutes"])

        has_cols = [c for c in ["goals_per90","assists_per90"] if c in df.columns]
        if has_cols:
            df["impact_index"] = df[has_cols].sum(axis=1).round(3)

        out = csv_path_metrics(PLAYER_METRICS)
        df.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"[OK] player metrics -> {out} ({len(df)} rows)")
        return df

    def build_match_metrics(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des flags et scores dérivés par match :
          - over_2_5 (Total Goals > 2.5)
          - both_teams_score (BTTS)
          - stage_weight (pondération phase)
        Export : metrics/match_metrics.csv
        """
        if matches_df is None or matches_df.empty:
            return pd.DataFrame()
        df = matches_df.copy()

        df["over_2_5"] = (pd.to_numeric(df.get("total_goals", np.nan), errors="coerce") > 2.5).astype("Int64")
        df["both_teams_score"] = ((df.get("home_goals", 0) > 0) & (df.get("away_goals", 0) > 0)).astype("Int64")

        # Pondérations de phases (utile pour storylines / ranking interne)
        phase_weight = {
            "Group Stage": 1, "Round Of 16": 2, "Round of 16": 2,
            "Quarter-Finals": 3, "Quarter-finals": 3, "Quarterfinals": 3,
            "Semi-Finals": 4, "Semi-finals": 4, "Semifinals": 4,
            "Final": 5
        }
        if "stage" in df.columns:
            df["stage_weight"] = df["stage"].map(lambda x: phase_weight.get(str(x), 1))
        else:
            df["stage_weight"] = 1

        out = csv_path_metrics(MATCH_METRICS)
        df.to_csv(out, index=False, encoding="utf-8")
        logger.info(f"[OK] match metrics -> {out} ({len(df)} rows)")
        return df

    # ---------------- Pipeline complet ----------------
    def compute_and_export_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Orchestration end-to-end :
          1) process_matches
          2) process_standings (+ fallback)
          3) process_player_ranking (tolérant)
          4) process_players (priorité Merged > Per-match > Ranking)
          5) process_teams (priorité standings)
          6) build_*_metrics (match / team / player)
        Renvoie (matches, standings, players, teams)
        """
        m = self.process_matches()
        s = self.process_standings()

        try:
            self.process_player_ranking()
        except Exception as e:
            logger.warning(f"[WARN] process_player_ranking a échoué: {e}")

        p = self.process_players()
        t = self.process_teams()

        self.build_match_metrics(m)
        self.build_team_metrics(m, s)
        self.build_player_metrics(p)

        return m, s, p, t


# ---------------- Entrée CLI simple ----------------
if __name__ == "__main__":
    dp = DataProcessor()
    dp.compute_and_export_all()
    print("✅ Processing & metrics done.")
