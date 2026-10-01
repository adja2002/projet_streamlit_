"""
Scraper UCL 2024/25 — API 'match.uefa.com' + fallback HTML (__NEXT_DATA__) + compstats (joueurs)
Sorties:
  - matches     -> data/raw/matches.csv
  - groups      -> data/raw/standings.csv
  - teamstats   -> data/raw/match_stats.csv
  - players_pm  -> data/raw/players_stats_per_match.csv
  - players     -> data/raw/players.csv (agrégé)

Dans ce module, je construis un pipeline de scraping robuste pour la saison UCL 2024/25.
Je combine 3 sources :
  1) API match.uefa.com (v3/v4/v5) pour la liste/détails,
  2) le HTML public de uefa.com (parse du JSON Next.js: __NEXT_DATA__) en fallback,
  3) l’API compstats.uefa.com pour un agrégat rapide des joueurs.
Je veille à :
  - gérer les timeouts/erreurs/retries,
  - normaliser au maximum les libellés (équipes, stades, villes),
  - écrire des CSV cohérents réutilisables dans le processing/les dashboards.
"""
from __future__ import annotations

import sys
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests
import pandas as pd

# Streamlit est optionnel (si lancé en CLI hors app Streamlit)
try:
    import streamlit as st  # type: ignore
except Exception:
    class _Dummy:
        def warning(self, *a, **k): print("WARNING:", *a)
        def success(self, *a, **k): print("SUCCESS:", *a)
        def error(self, *a, **k): print("ERROR:", *a)
        def info(self, *a, **k): print("INFO:", *a)
        def dataframe(self, *a, **k): pass
    st = _Dummy()

# -- chemins projet : je rends le dossier racine importable et j'importe la config des chemins CSV
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import csv_path_raw, MATCHES_FILE, STANDINGS_FILE, PLAYERS_FILE  # noqa: E402

# ---------------- Constantes HTTP ----------------
# En-têtes "navigateur" pour éviter d'être bloquée par certains endpoints
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.uefa.com/",
}
# Liste des endpoints : la liste (pagination) et plusieurs versions pour le détail (v5 -> v4 -> v3)
MATCH_LIST_URL = "https://match.uefa.com/v5/matches"
MATCH_DETAIL_CANDIDATES = [
    "https://match.uefa.com/v5/matches/{mid}",
    "https://match.uefa.com/v4/matches/{mid}",
    "https://match.uefa.com/v3/matches/{mid}",
]
UEFA_HOST = "https://www.uefa.com"
# Slugs HTML utiles où je peux trouver le __NEXT_DATA__ (fallback)
_HTML_SLUGS = [
    "/uefachampionsleague/match/{mid}/statistics/",
    "/uefachampionsleague/match/{mid}/lineups/",
    "/uefachampionsleague/match/{mid}/timeline/",
    "/uefachampionsleague/match/{mid}/matchinfo/",
    "/uefachampionsleague/match/{mid}/",
]

# ---------------- Petites utilitaires ----------------
def _first(*vals):
    """Je renvoie la première valeur non vide/non nulle utile."""
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
        if v not in (None, "", {}):
            return v
    return None

def _tval(d: Dict[str, Any] | None, path: List[str]) -> Optional[Any]:
    """Accès sécurisé à une clé profonde : je parcours 'path' dans le dict 'd'."""
    cur = d
    try:
        for k in path:
            if cur is None or not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur
    except Exception:
        return None

def _extract_trans_name(obj: Dict[str, Any] | None, key: str = "name") -> Optional[str]:
    """Certains objets ont 'translations' → je récupère FR puis EN en fallback."""
    if not isinstance(obj, dict):
        return None
    tr = obj.get("translations") or {}
    kk = tr.get(key) or {}
    return _first(kk.get("EN"), kk.get("FR"))

def _tname(team: Dict[str, Any] | None) -> str:
    """Nom d’équipe homogène : j’essaie displayName/name FR/EN, puis d’autres champs, sinon 'Inconnu'."""
    if not isinstance(team, dict):
        return "Inconnu"
    tr = team.get("translations") or {}
    dn = tr.get("displayName") or {}
    nm = tr.get("name") or {}
    return (
        _first(
            dn.get("EN"),
            dn.get("FR"),
            nm.get("EN"),
            nm.get("FR"),
            team.get("internationalName"),
            team.get("officialName"),
            team.get("shortName"),
            team.get("name"),
        )
        or "Inconnu"
    )

def _deep_find(obj: Any, keys: List[str]) -> List[dict]:
    """
    Recherche profonde : je parcours récursivement dict/list et je collecte
    les sous-dicts qui contiennent TOUTES les clés 'keys'.
    """
    out: List[dict] = []
    if isinstance(obj, dict):
        if all(k in obj for k in keys):
            out.append(obj)
        for v in obj.values():
            out.extend(_deep_find(v, keys))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_deep_find(it, keys))
    return out

# ---------------- Session HTTP avec retry ----------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _mk_session() -> requests.Session:
    """
    Je crée une session requests avec stratégie de retry :
    - total=5, backoff exponentiel
    - retries sur 429/5xx
    - GET uniquement
    """
    s = requests.Session()
    s.headers.update(_HEADERS)  # (Correction : _HEADERS et non __HEADERS)
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

# ---------------- Fetchers de base ----------------
def _fetch_match_detail(mid: int) -> Optional[Dict[str, Any]]:
    """
    J’essaie plusieurs versions de l’API détail (v5→v4→v3) pour maximiser les chances.
    """
    for fmt in MATCH_DETAIL_CANDIDATES:
        url = fmt.format(mid=mid)
        try:
            r = requests.get(url, headers=_HEADERS, timeout=15)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(0.2)
            continue
    return None

def _fetch_next_data(mid: int) -> Optional[Dict[str, Any]]:
    """
    Fallback HTML : je télécharge la page publique et je parse le JSON '__NEXT_DATA__'
    rendu côté serveur par Next.js (uefa.com).
    """
    for slug in _HTML_SLUGS:
        url = f"{UEFA_HOST}{slug.format(mid=mid)}"
        try:
            r = requests.get(
                url,
                headers={
                    "User-Agent": _HEADERS["User-Agent"],
                    "Accept": "text/html,*/*",
                    "Referer": "https://www.uefa.com/",
                },
                timeout=20,
            )
            if r.status_code != 200 or not r.text:
                continue
            import re as _re
            import json as _json

            m = _re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
                r.text,
                _re.S,
            )
            if not m:
                continue
            return _json.loads(m.group(1))
        except Exception:
            time.sleep(0.15)
            continue
    return None

# ---------------- Scrape: Matches (liste + enrichissement détail) ----------------
def scrape_matches(limit_per_page: int = 50, verbose: bool = True) -> pd.DataFrame:
    """
    Je parcours /v5/matches (paginé) pour récupérer toute la saison (phase=TOURNAMENT).
    Pour chaque match, je tente un enrichissement via le détail (v5/v4/v3) :
      - date, groupe, pays/stade/ville, arbitre…
    J’écris ensuite le CSV consolidé.
    """
    all_rows: List[Dict[str, Any]] = []
    offset, page, seen_ids = 0, 0, set()
    sess = _mk_session()
    if verbose:
        print(f"[matches] GET {MATCH_LIST_URL} ...")
    while True:
        page += 1
        params = {
            "competitionId": 1,
            "seasonYear": 2024,
            "phase": "TOURNAMENT",
            "order": "ASC",
            "limit": limit_per_page,
            "offset": offset,
        }
        try:
            r = sess.get(MATCH_LIST_URL, params=params, timeout=15)
            if verbose:
                print(f"[matches] page {page} status={r.status_code} url={r.url}")
            r.raise_for_status()
            try:
                batch = r.json()
            except Exception as e:
                print(f"[matches] JSON decode error: {e}")
                break
            if not batch:
                if verbose:
                    print(f"[matches] Fin: batch vide (offset={offset})")
                break

            # Je normalise chaque entrée
            for m in batch:
                mid = m.get("id")
                if not mid or mid in seen_ids:
                    continue
                seen_ids.add(mid)

                ko = m.get("kickOffTime") or {}
                stadium_obj = m.get("stadium") or {}
                city_obj = stadium_obj.get("city") or {}
                row = {
                    "match_id": mid,
                    "utc_date": (ko.get("dateTime")),
                    "stage": _tval(m, ["round", "metaData", "name"]),
                    "round_code": _tval(m, ["round", "id"]),
                    "home_team": _tname(m.get("homeTeam")),
                    "away_team": _tname(m.get("awayTeam")),
                    "home_score": _tval(m, ["score", "regular", "home"]) or 0,
                    "away_score": _tval(m, ["score", "regular", "away"]) or 0,
                    "stadium": _extract_trans_name(stadium_obj, "name") or "Inconnu",
                    "city": _extract_trans_name(city_obj, "name") or "Inconnu",
                    "attendance": m.get("matchAttendance") or 0,
                    "status": m.get("status"),
                    "season": None,
                    "group": None,
                    "country": _extract_trans_name(stadium_obj.get("country") or {}, "name"),
                    "referee": None,
                }

                # Enrichissement détail si dispo
                detail = _fetch_match_detail(mid)
                if detail:
                    date_iso2 = _tval(detail, ["kickOffTime", "dateTime"])
                    row["utc_date"] = _first(row["utc_date"], date_iso2)
                    row["season"] = _first(
                        detail.get("seasonYear"),
                        _tval(detail, ["season", "seasonYear"]),
                        _tval(detail, ["competition", "season", "year"]),
                    )
                    grp = detail.get("group") or {}
                    row["group"] = _first(
                        _extract_trans_name(grp, "name"),
                        _extract_trans_name(grp, "shortName"),
                        grp.get("groupName"),
                        grp.get("name"),
                    )
                    if not row["country"]:
                        row["country"] = _extract_trans_name(
                            _tval(detail, ["stadium", "country"]) or {}, "name"
                        )
                    row["referee"] = _extract_referee(detail)
                    if row["stadium"] in (None, "Inconnu"):
                        row["stadium"] = (
                            _extract_trans_name(_tval(detail, ["stadium"]) or {}, "name")
                            or row["stadium"]
                        )
                    if row["city"] in (None, "Inconnu"):
                        row["city"] = (
                            _extract_trans_name(_tval(detail, ["stadium", "city"]) or {}, "name")
                            or row["city"]
                        )

                all_rows.append(row)
                time.sleep(0.10)  # petit sleep pour ne pas spammer
            offset += limit_per_page
            time.sleep(0.20)
        except requests.exceptions.RequestException as e:
            print(f"[matches] HTTP error: {e}")
            break

    # DataFrame + enrichissements dérivés (dates, résultats)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["match_id"])
    if "utc_date" in df.columns:
        df["date_utc"] = pd.to_datetime(df["utc_date"], errors="coerce", utc=True)
        df["date"] = df["date_utc"]
    for c in ["home_score", "away_score", "attendance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "date_utc" in df.columns:
        df["year"] = df["date_utc"].dt.year
        df["month"] = df["date_utc"].dt.month
        df["weekday"] = df["date_utc"].dt.weekday
    if not df.empty:
        def _res(r):
            if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
                return None
            return "H" if r["home_score"] > r["away_score"] else ("A" if r["home_score"] < r["away_score"] else "D")
        df["result"] = df.apply(_res, axis=1)
        df["home_goals"] = df["home_score"]
        df["away_goals"] = df["away_score"]

    # Colonnes en ordre lisible
    wanted = [
        "match_id", "utc_date", "stage", "round_code", "home_team", "away_team",
        "home_score", "away_score", "stadium", "city", "attendance", "status",
        "season", "group", "country", "referee", "date_utc", "date", "year",
        "month", "weekday", "result", "home_goals", "away_goals",
    ]
    df = df[[c for c in wanted if c in df.columns] + [c for c in df.columns if c not in wanted]].sort_values("date_utc")

    # Export CSV
    out_local = csv_path_raw(MATCHES_FILE)
    out_local.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_local, index=False, encoding="utf-8")
    print(f"[matches] saved -> {out_local} ({len(df)} rows)")
    return df

def _extract_referee(j: Dict[str, Any]) -> Optional[str]:
    """
    J’essaie plusieurs structures possibles pour retrouver l’arbitre principal.
    """
    keys = ["officials", "matchOfficials", "referees", "matchOfficialsList"]
    for k in keys:
        arr = j.get(k)
        if isinstance(arr, list):
            for o in arr:
                role = o.get("role") or o.get("function") or {}
                role_code = (role.get("code") if isinstance(role, dict) else role) or ""
                role_name = (role.get("name") if isinstance(role, dict) else role) or ""
                if "ref" in f"{role_code} {role_name}".lower():
                    person = o.get("person") or o.get("official") or o
                    nm = (
                        _extract_trans_name(person, "name")
                        or person.get("displayName")
                        or person.get("name")
                    )
                    if nm:
                        return nm
    ref = j.get("referee")
    if isinstance(ref, dict):
        return _extract_trans_name(ref, "name") or ref.get("name")
    if isinstance(ref, str):
        return ref
    return None

# ---------------- Scrape: Groups / Standings ----------------
def scrape_groups(verbose: bool = True) -> pd.DataFrame:
    """
    Je consomme standings.uefa.com pour reconstituer les classements de groupes.
    Je normalise les noms d’équipes et j’extrais les stats principales (points, W/D/L, GF/GA…).
    """
    try:
        url = "https://standings.uefa.com/v1/standings?competitionId=1&seasonYear=2024"
        if verbose:
            print(f"[groups] GET {url} ...")
        r = requests.get(url, headers=_HEADERS, timeout=15)
        print(f"[groups] status={r.status_code}")
        r.raise_for_status()
        data = r.json()
        group_list = data if isinstance(data, list) else (data.get("standings") or data.get("groups") or [])

        def norm_group_name(group_obj):
            if isinstance(group_obj, dict):
                return (
                    _extract_trans_name(group_obj, "name")
                    or _extract_trans_name(group_obj, "shortName")
                    or group_obj.get("groupName")
                    or group_obj.get("name")
                    or group_obj.get("id")
                    or "Groupe inconnu"
                )
            return str(group_obj) if group_obj is not None else "Groupe inconnu"

        def team_display_name(team):
            if not isinstance(team, dict):
                return "Équipe inconnue"
            tr = team.get("translations") or {}
            return (
                _first(
                    (tr.get("name") or {}).get("FR"),
                    (tr.get("displayName") or {}).get("FR"),
                    (tr.get("name") or {}).get("EN"),
                    (tr.get("displayName") or {}).get("EN"),
                    team.get("internationalName"),
                    team.get("officialName"),
                    team.get("shortName"),
                    team.get("name"),
                )
                or "Équipe inconnue"
            )

        def get_stat(item, key):
            """Utilitaire : une stat peut être directe ou dans item['stats'] avec (name,value)."""
            if not isinstance(item, dict):
                return 0
            v = item.get(key)
            if v is None:
                stats = item.get("stats")
                if isinstance(stats, list):
                    for s in stats:
                        if s.get("name") == key and s.get("value") is not None:
                            v = s["value"]
                            break
            try:
                return int(v) if v is not None else 0
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return 0

        rows = []
        for g in group_list:
            if not isinstance(g, dict):
                continue
            group_name = norm_group_name(g.get("group"))
            items = g.get("items") or g.get("teams") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                team_info = item.get("team") or {}
                rows.append(
                    {
                        "group": group_name,
                        "team": team_display_name(team_info),
                        "points": get_stat(item, "points"),
                        "wins": get_stat(item, "won"),
                        "draws": get_stat(item, "drawn"),
                        "losses": get_stat(item, "lost"),
                        "goals_for": get_stat(item, "goalsFor"),
                        "goals_against": get_stat(item, "goalsAgainst"),
                        "goal_difference": get_stat(item, "goalDifference"),
                        "played": get_stat(item, "played"),
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            # Fallback neutre pour que le reste du pipeline n’explose pas
            print("[groups] Aucune ligne reçue — fallback vierge.")
            df = pd.DataFrame(
                [
                    {
                        "group": f"Groupe {g}",
                        "team": f"Équipe {i+1}",
                        "points": 0,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "goals_for": 0,
                        "goals_against": 0,
                        "goal_difference": 0,
                        "played": 0,
                    }
                    for g in "ABCDEFGH"
                    for i in range(4)
                ]
            )
        out = csv_path_raw(STANDINGS_FILE)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[groups] saved -> {out} ({len(df)} rows)")
        return df
    except requests.exceptions.RequestException as e:
        print(f"[groups] HTTP error: {e}")
        return pd.DataFrame()

# ---------------- Team stats par match ----------------
def extract_team_stats_from_detail(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    À partir du JSON détail d’un match, j’extrais les statistiques d’équipe (home/away) en long :
      match_id, team_id, team_name, stat_key, stat_value
    Je recherche différentes formes possibles (statistics, matchStats, stats).
    """
    rows: List[Dict[str, Any]] = []
    match_id = _tval(detail, ["id"]) or _tval(detail, ["matchId"])
    cands: List[dict] = []
    cands += _deep_find(detail, ["teams", "statistics"])
    cands += _deep_find(detail, ["teams", "matchStats"])
    cands += _deep_find(detail, ["homeTeam", "awayTeam", "statistics"])
    for node in cands:
        # Je reconstitue la liste des deux équipes (selon structures possibles)
        teams = node.get("teams")
        if not isinstance(teams, list):
            ht, at = node.get("homeTeam") or {}, node.get("awayTeam") or {}
            teams = [{"id": ht.get("id"), "name": _tname(ht)}, {"id": at.get("id"), "name": _tname(at)}]
        if not teams or len(teams) < 2:
            continue
        stats = node.get("statistics") or node.get("matchStats") or node.get("stats") or []
        if not isinstance(stats, list) or not stats:
            continue
        # Je dédouble (home/away) pour chaque clé statistique
        for it in stats:
            key = (it.get("name") or it.get("label") or "").strip()
            home = it.get("home") or it.get("valueHome") or it.get("homeValue")
            away = it.get("away") or it.get("valueAway") or it.get("awayValue")
            if isinstance(home, dict) and "value" in home:
                home = home.get("value")
            if isinstance(away, dict) and "value" in away:
                away = away.get("value")
            if not key or (home is None and away is None):
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": teams[0].get("id"),
                    "team_name": teams[0].get("name") or _tname(teams[0]),
                    "stat_key": key,
                    "stat_value": home,
                }
            )
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": teams[1].get("id"),
                    "team_name": teams[1].get("name") or _tname(teams[1]),
                    "stat_key": key,
                    "stat_value": away,
                }
            )
        if rows:
            break
    return rows

def _extract_team_stats_from_nextdata(nd: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Même extraction que ci-dessus, mais à partir du __NEXT_DATA__ (fallback HTML).
    """
    rows: List[Dict[str, Any]] = []
    match_id = None
    for cand in _deep_find(nd, ["matchId"]):
        if isinstance(cand.get("matchId"), (int, str)):
            match_id = cand.get("matchId")
            break
    if match_id is None:
        pg = nd.get("props", {}).get("pageProps", {})
        match_id = pg.get("matchId") or pg.get("id")
    candidates: List[dict] = []
    candidates += _deep_find(nd, ["teams", "statistics"])
    candidates += _deep_find(nd, ["teams", "matchStats"])
    candidates += _deep_find(nd, ["homeTeam", "awayTeam", "statistics"])
    for node in candidates:
        teams = node.get("teams")
        if not isinstance(teams, list):
            ht, at = node.get("homeTeam") or {}, node.get("awayTeam") or {}
            teams = [{"id": ht.get("id"), "name": _tname(ht)}, {"id": at.get("id"), "name": _tname(at)}]
        if not teams or len(teams) < 2:
            continue
        stats = node.get("statistics") or node.get("matchStats") or node.get("stats") or []
        if not isinstance(stats, list) or not stats:
            continue
        for it in stats:
            key = (it.get("name") or it.get("label") or "").strip()
            home = it.get("home") or it.get("valueHome") or it.get("homeValue")
            away = it.get("away") or it.get("valueAway") or it.get("awayValue")
            if isinstance(home, dict) and "value" in home:
                home = home.get("value")
            if isinstance(away, dict) and "value" in away:
                away = away.get("value")
            if not key or (home is None and away is None):
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": teams[0].get("id"),
                    "team_name": teams[0].get("name") or _tname(teams[0]),
                    "stat_key": key,
                    "stat_value": home,
                }
            )
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": teams[1].get("id"),
                    "team_name": teams[1].get("name") or _tname(teams[1]),
                    "stat_key": key,
                    "stat_value": away,
                }
            )
        if rows:
            break
    return rows

def scrape_team_stats(from_matches_csv: str = "data/raw/matches.csv",
                      out_csv: str = "data/raw/match_stats.csv") -> None:
    """
    Orchestrateur des stats d’équipes :
      - je lis la liste des matchs (CSV),
      - pour chaque match : API détail puis fallback HTML,
      - j’écris le long format 'match_stats.csv'.
    """
    dfm = pd.read_csv(from_matches_csv)
    if "match_id" not in dfm.columns:
        raise SystemExit("Le CSV des matchs doit contenir 'match_id'")
    all_rows: List[Dict[str, Any]] = []
    for _, r in dfm.iterrows():
        mid = int(r["match_id"])
        used_rows: List[Dict[str, Any]] = []
        detail = _fetch_match_detail(mid)
        if detail:
            used_rows = extract_team_stats_from_detail(detail)
        if not used_rows:
            nd = _fetch_next_data(mid)
            if nd:
                used_rows = _extract_team_stats_from_nextdata(nd)
        if used_rows:
            all_rows.extend(used_rows)
        time.sleep(0.10)
    cols = ["match_id", "team_id", "team_name", "stat_key", "stat_value"]
    pd.DataFrame(all_rows, columns=cols).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[teamstats] saved -> {out_csv} ({len(all_rows)} lignes)")

# ---------------- Players par match + agrégé ----------------
def _collect_lineups(detail: Dict[str, Any]):
    """
    Je tente de reconstruire les compositions (titulaires/remplaçants) selon
    plusieurs structures possibles ('lineups', 'squads', 'players').
    """
    teams_out: List[Dict[str, Any]] = []
    lineup_nodes: List[dict] = []
    lineup_nodes += _deep_find(detail, ["lineups"])
    lineup_nodes += _deep_find(detail, ["squads"])
    lineup_nodes += _deep_find(detail, ["players", "team"])
    for ln in lineup_nodes:
        arr = ln.get("lineups") or ln.get("squads") or []
        if isinstance(arr, list) and arr:
            # Structure lineups/squads classique
            for side in arr:
                tid = side.get("teamId") or _tval(side, ["team", "id"]) or side.get("id")
                tname = side.get("teamName") or _tname(side.get("team") or {}) or side.get("name")
                starters = side.get("startingXI") or side.get("starters") or []
                subs = side.get("substitutes") or side.get("subs") or []
                players: List[Dict[str, Any]] = []
                for p in starters:
                    pid = p.get("id") or p.get("playerId")
                    pname = p.get("name") or p.get("playerName")
                    if pid or pname:
                        players.append({"player_id": pid, "player_name": pname, "starter": True})
                for p in subs:
                    pid = p.get("id") or p.get("playerId")
                    pname = p.get("name") or p.get("playerName")
                    if pid or pname:
                        players.append({"player_id": pid, "player_name": pname, "starter": False})
                if tname and players:
                    teams_out.append({"team_id": tid, "team_name": tname, "players": players})
            if teams_out:
                return teams_out
        # Variante : liste 'players' déjà aplatie avec référence à l'équipe
        players = ln.get("players")
        if isinstance(players, list) and players:
            by_team: Dict[str, Dict[str, Any]] = {}
            for p in players:
                team = p.get("team") or {}
                tid = team.get("id")
                tname = _tname(team)
                pname = p.get("name") or p.get("playerName")
                pid = p.get("id") or p.get("playerId")
                starter = bool(p.get("isStarter") or p.get("starter"))
                if not tname or not (pid or pname):
                    continue
                by_team.setdefault(tname, {"team_id": tid, "team_name": tname, "players": []})
                by_team[tname]["players"].append({"player_id": pid, "player_name": pname, "starter": starter})
            if by_team:
                return list(by_team.values())
    return teams_out

def _collect_events(detail: Dict[str, Any]):
    """
    J’essaie de récupérer la timeline d’un match (événements) selon différentes clés.
    """
    ev_nodes: List[dict] = []
    ev_nodes += _deep_find(detail, ["events"])
    ev_nodes += _deep_find(detail, ["matchEvents"])
    ev_nodes += _deep_find(detail, ["timeline"])
    for node in ev_nodes:
        arr = node.get("events") or node.get("matchEvents") or node.get("timeline") or []
        if isinstance(arr, list) and arr:
            return arr
    return []

def extract_player_rows_from_detail(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    À partir d’un 'detail' de match, je produis une ligne par joueur et par match :
      minutes jouées (approx. via subs), buts, assists, cartons.
    Hypothèses :
      - durée 90' (120' si prolongations détectées),
      - starter=True → commence à 0', sort si 'sub_out',
      - remplaçant → entre à 'sub_in', sinon minutes=0.
    """
    match_id = _tval(detail, ["id"]) or _tval(detail, ["matchId"])
    duration = _tval(detail, ["duration"]) or 90
    extra = bool(_tval(detail, ["extraTime"])) or False
    try:
        duration = int(duration)
    except Exception:
        duration = 90
    if extra and duration < 120:
        duration = 120

    teams = _collect_lineups(detail)
    events = _collect_events(detail)

    # Dictionnaires d’accumulation par player_id
    goals, assists, yellow, red, sub_in, sub_out = {}, {}, {}, {}, {}, {}
    def _inc(d, k, v=1): d[k] = d.get(k, 0) + v

    # Je parcours les événements et j’incrémente / infère les substitutions
    for ev in events:
        etype = (ev.get("type") or ev.get("eventType") or "").lower()
        minute = ev.get("minute") or ev.get("time") or ev.get("eventMinute") or ev.get("clock")
        try:
            minute = int(minute)
        except Exception:
            minute = None
        pid = ev.get("playerId") or ev.get("player") or ev.get("pid")
        aid = ev.get("assistId") or ev.get("assist")
        if etype in ("goal", "but", "goalnormal", "penaltyscored", "owngoal"):
            if pid: _inc(goals, pid)
            if aid: _inc(assists, aid)
        elif etype in ("yellow", "yellowcard", "yc"):
            if pid: _inc(yellow, pid)
        elif etype in ("red", "redcard", "rc"):
            if pid: _inc(red, pid)
        elif etype in ("substitution_on", "subon", "in", "subonplayer"):
            if pid and minute is not None: sub_in[pid] = min(minute, duration)
        elif etype in ("substitution_off", "suboff", "out", "suboffplayer"):
            if pid and minute is not None: sub_out[pid] = min(minute, duration)

    # Construction des lignes joueurs
    rows: List[Dict[str, Any]] = []
    for t in teams:
        tid, tname = t["team_id"], t["team_name"]
        for p in t["players"]:
            pid, pname, starter = p.get("player_id"), p.get("player_name"), bool(p.get("starter"))
            if starter:
                start_min = 0
                end_min = sub_out.get(pid, duration)
            else:
                start_min = sub_in.get(pid, duration)
                end_min = duration if pid in sub_in else start_min
            minutes = max(0, (end_min or 0) - (start_min or 0))
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": tid,
                    "team_name": tname,
                    "player_id": pid,
                    "player_name": pname,
                    "starter": starter,
                    "minutes": int(minutes),
                    "goals": int(goals.get(pid, 0)),
                    "assists": int(assists.get(pid, 0)),
                    "yellow": int(yellow.get(pid, 0)),
                    "red": int(red.get(pid, 0)),
                }
            )
    return rows

def _extract_player_rows_from_nextdata(nd: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Variante de l’extraction joueurs en s’appuyant sur le __NEXT_DATA__.
    Je reconstruis un 'detail_like' minimal pour réutiliser la même logique.
    """
    match_id = None
    for cand in _deep_find(nd, ["matchId"]):
        if isinstance(cand.get("matchId"), (int, str)):
            match_id = cand.get("matchId")
            break
    if match_id is None:
        pg = nd.get("props", {}).get("pageProps", {})
        match_id = pg.get("matchId") or pg.get("id")

    # Reconstruction des lineups
    teams = []
    lineup_nodes = []
    lineup_nodes += _deep_find(nd, ["lineups"])
    lineup_nodes += _deep_find(nd, ["squads"])
    lineup_nodes += _deep_find(nd, ["players", "team"])

    def _mk_team(side):
        tid = side.get("teamId") or _tval(side, ["team", "id"]) or side.get("id")
        tname = side.get("teamName") or _tname(side.get("team") or {}) or side.get("name")
        starters = side.get("startingXI") or side.get("starters") or []
        subs = side.get("substitutes") or side.get("subs") or []
        players = []
        for p in starters:
            pid = p.get("id") or p.get("playerId")
            pname = p.get("name") or p.get("playerName")
            if pid or pname:
                players.append({"player_id": pid, "player_name": pname, "starter": True})
        for p in subs:
            pid = p.get("id") or p.get("playerId")
            pname = p.get("name") or p.get("playerName")
            if pid or pname:
                players.append({"player_id": pid, "player_name": pname, "starter": False})
        if tname and players:
            return {"team_id": tid, "team_name": tname, "players": players}
        return None

    found = False
    for ln in lineup_nodes:
        arr = ln.get("lineups") or ln.get("squads") or []
        if isinstance(arr, list) and arr:
            for side in arr:
                tt = _mk_team(side)
                if tt:
                    teams.append(tt)
            if teams:
                found = True
                break
        players = ln.get("players")
        if isinstance(players, list) and players:
            by_team: Dict[str, Dict[str, Any]] = {}
            for p in players:
                team = p.get("team") or {}
                tid = team.get("id")
                tname = _tname(team)
                pname = p.get("name") or p.get("playerName")
                pid = p.get("id") or p.get("playerId")
                starter = bool(p.get("isStarter") or p.get("starter"))
                if not tname or not (pid or pname):
                    continue
                by_team.setdefault(tname, {"team_id": tid, "team_name": tname, "players": []})
                by_team[tname]["players"].append({"player_id": pid, "player_name": pname, "starter": starter})
            if by_team:
                teams = list(by_team.values())
                found = True

    # Événements
    events = []
    for node in _deep_find(nd, ["events"]) + _deep_find(nd, ["matchEvents"]) + _deep_find(nd, ["timeline"]):
        arr = node.get("events") or node.get("matchEvents") or node.get("timeline") or []
        if isinstance(arr, list) and arr:
            events = arr
            break

    # Je fabrique un 'detail_like' pour réutiliser extract_player_rows_from_detail
    duration = 120 if any(d.get("extraTime") for d in _deep_find(nd, ["extraTime"])) else 90
    detail_like: Dict[str, Any] = {"id": match_id, "duration": duration, "lineups": [{"lineups": []}]}
    if found:
        detail_like["lineups"] = [{
            "lineups": [{
                "teamId": t["team_id"],
                "teamName": t["team_name"],
                "startingXI": [{"id": p["player_id"], "name": p["player_name"]} for p in t["players"] if p["starter"]],
                "substitutes": [{"id": p["player_id"], "name": p["player_name"]} for p in t["players"] if not p["starter"]],
            } for t in teams]
        }]
    if events:
        detail_like["events"] = {"events": events}
    return extract_player_rows_from_detail(detail_like)

def scrape_player_stats(from_matches_csv: str = "data/raw/matches.csv",
                        out_csv: str = "data/raw/players_stats_per_match.csv",
                        aggregate_out: Optional[str] = None) -> None:
    """
    Orchestrateur joueurs :
      - j’extrais joueur/match (minutes, buts, passes, cartons),
      - j’écris le CSV par match,
      - j’agrège par joueur+saison vers players.csv (cols extra à None pour enrichissements futurs).
    """
    if aggregate_out is None:
        aggregate_out = csv_path_raw(PLAYERS_FILE)

    dfm = pd.read_csv(from_matches_csv)
    if "match_id" not in dfm.columns:
        raise SystemExit("Le CSV des matchs doit contenir 'match_id'")

    all_rows: List[Dict[str, Any]] = []
    for _, r in dfm.iterrows():
        mid = int(r["match_id"])
        rows: List[Dict[str, Any]] = []
        detail = _fetch_match_detail(mid)
        if detail:
            rows = extract_player_rows_from_detail(detail)
        if not rows:
            nd = _fetch_next_data(mid)
            if nd:
                rows = _extract_player_rows_from_nextdata(nd)
        if rows:
            all_rows.extend(rows)
        time.sleep(0.10)

    cols = [
        "match_id", "team_id", "team_name", "player_id", "player_name", "starter",
        "minutes", "goals", "assists", "yellow", "red",
    ]
    df_pm = pd.DataFrame(all_rows, columns=cols)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_pm.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[players] per-match saved -> {out_csv} ({len(df_pm)} lignes)")

    # Agrégation saison -> players.csv
    if df_pm.empty:
        print("[players] per-match vide — agrégé vide.")
        pd.DataFrame(
            columns=[
                "category", "player_name", "team", "position", "goals", "assists", "minutes",
                "yellow", "red", "matches", "xg", "xa", "shots", "key_passes", "dribbles_won", "tackles", "age",
            ]
        ).to_csv(aggregate_out, index=False, encoding="utf-8")
        print(f"[players] aggregated saved -> {aggregate_out} (0 lignes)")
        return

    # Nettoyage simple des noms
    df_pm["player_name"] = df_pm["player_name"].astype(str).str.strip()
    df_pm["team_name"] = df_pm["team_name"].astype(str).str.strip()

    # Agrégat final par joueur/équipe
    agg = (
        df_pm.groupby(["player_name", "team_name"], as_index=False)
            .agg(
                minutes=("minutes", "sum"),
                goals=("goals", "sum"),
                assists=("assists", "sum"),
                yellow=("yellow", "sum"),
                red=("red", "sum"),
                matches=("match_id", "nunique"),
            )
    )
    agg = agg.rename(columns={"team_name": "team"})
    agg["category"] = "UCL"
    # Colonnes prévues pour enrichissements ultérieurs (xG/xa/shots/age/position…)
    agg["position"] = None
    agg["xg"] = None
    agg["xa"] = None
    agg["shots"] = None
    agg["key_passes"] = None
    agg["dribbles_won"] = None
    agg["tackles"] = None
    agg["age"] = None

    ordered = [
        "category", "player_name", "team", "position", "goals", "assists", "minutes",
        "yellow", "red", "matches", "xg", "xa", "shots", "key_passes", "dribbles_won", "tackles", "age",
    ]
    agg = agg[ordered]

    Path(aggregate_out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(aggregate_out, index=False, encoding="utf-8")
    print(f"[players] aggregated saved -> {aggregate_out} ({len(agg)} lignes)")

# ---------------- Players (compstats, rapide) ----------------
def _stat_pick(bag, key: str) -> int:
    """Je récupère une stat 'key' dans une liste de dicts (name/value)."""
    if not isinstance(bag, list):
        return 0
    for it in bag:
        if isinstance(it, dict) and str(it.get("name")).lower() == key.lower():
            try:
                v = it.get("value", 0)
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return 0
    return 0

def _p_label(p: dict) -> str:
    """Libellé joueur “propre” (affichage/nom international)."""
    return (p.get("displayName") or p.get("internationalName") or p.get("name") or "Joueur inconnu")

def _t_label(t: dict) -> str:
    """Libellé équipe “propre” (shortName/officialName)."""
    return (t.get("officialName") or t.get("internationalName") or t.get("shortName") or t.get("name") or "Équipe inconnue")

def scrape_players_compstats_fast(season_year: int = 2024,
                                  phase: str = "TOURNAMENT",
                                  page_size: int = 200,
                                  max_pages: int = 20) -> pd.DataFrame:
    """
    Récupération rapide 'player-ranking' (compstats) :
      - minutes, apparitions, buts, passes, jaunes, rouges
      - j’écris directement players.csv (agrégé)
    Remarque : cette source est pratique pour démarrer quand le parsing par match est long.
    """
    try:
        base_url = "https://compstats.uefa.com/v1/player-ranking"
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.uefa.com/",
        })
        wanted = ["minutes_played_official", "matches_appearance", "goals", "assists", "yellow_cards", "red_cards"]
        all_rows = []
        seen_keys = set()
        for page in range(max_pages):
            offset = page * page_size
            params = {
                "competitionId": 1,
                "limit": page_size,
                "offset": offset,
                "optionalFields": "PLAYER,TEAM",
                "order": "DESC",
                "phase": phase,
                "seasonYear": season_year,
                "stats": ",".join(wanted),
            }
            r = sess.get(base_url, params=params, timeout=20)
            if r.status_code == 404:
                break
            r.raise_for_status()
            payload = r.json()
            if not payload:
                break
            batch = 0
            for item in payload if isinstance(payload, list) else []:
                player = item.get("player", {}) or {}
                team = item.get("team", {}) or {}
                stats = item.get("statistics", []) or []
                key_tuple = (player.get("id"), _p_label(player), _t_label(team))
                if key_tuple in seen_keys:
                    continue
                seen_keys.add(key_tuple)
                row = {
                    "category": "UCL",
                    "player_name": _p_label(player),
                    "team": _t_label(team),
                    "position": None,
                    "goals": _stat_pick(stats, "goals"),
                    "assists": _stat_pick(stats, "assists"),
                    "minutes": _stat_pick(stats, "minutes_played_official"),
                    "yellow": _stat_pick(stats, "yellow_cards"),
                    "red": _stat_pick(stats, "red_cards"),
                    "matches": _stat_pick(stats, "matches_appearance"),
                    "xg": None,
                    "xa": None,
                    "shots": None,
                    "key_passes": None,
                    "dribbles_won": None,
                    "tackles": None,
                    "age": None,
                }
                all_rows.append(row)
                batch += 1
            if batch < page_size:
                break
        df = pd.DataFrame(all_rows)
        out_csv = csv_path_raw(PLAYERS_FILE)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        try:
            st.success(f"[players-fast] {len(df)} joueurs enregistrés -> {out_csv}")
            if not df.empty:
                st.dataframe(df.head(12), use_container_width=True)
        except Exception:
            pass
        return df
    except Exception as e:
        try:
            st.error(f"[players-fast] erreur: {e}")
        except Exception:
            print(f"[players-fast] erreur: {e}")
        return pd.DataFrame()

# ================== Orchestrateur pratique ==================
def scrape_all(limit_per_page: int = 50,
               season_year: int = 2024,
               phase: str = "TOURNAMENT") -> None:
    """
    Enchaîne tous les scrapers (pour un run “one-click”) :
      1) matchs
      2) classements de groupes
      3) stats d’équipe par match
      4) joueurs (par match + agrégé)
      5) joueurs (compstats rapide)
    """
    try:
        st.info("Scraping MATCHES…")
    except Exception:
        pass
    df_m = scrape_matches(limit_per_page=limit_per_page, verbose=True)

    try:
        st.info("Scraping GROUPS…")
    except Exception:
        pass
    scrape_groups(verbose=True)

    try:
        st.info("Scraping TEAM STATS…")
    except Exception:
        pass
    if not df_m.empty:
        scrape_team_stats(from_matches_csv=csv_path_raw(MATCHES_FILE),
                          out_csv=csv_path_raw("match_stats.csv"))

    # Joueurs : par match + agrégé
    try:
        st.info("Scraping PLAYERS (per-match)…")
    except Exception:
        pass
    scrape_player_stats(from_matches_csv=csv_path_raw(MATCHES_FILE),
                        out_csv=csv_path_raw("players_stats_per_match.csv"),
                        aggregate_out=csv_path_raw(PLAYERS_FILE))

    # Joueurs (compstats rapide)
    try:
        st.info("Scraping PLAYERS (compstats fast)…")
    except Exception:
        pass
    scrape_players_compstats_fast(season_year=season_year, phase=phase)

# ================== CLI ==================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Scraper UCL 2024/25 — matches, groups, team stats, players."
    )
    ap.add_argument("what", choices=[
        "matches", "groups", "teamstats",
        "players_pm", "players_fast", "all"
    ])
    ap.add_argument("--limit", type=int, default=50, help="Page size pour /v5/matches")
    ap.add_argument("--season-year", type=int, default=2024, help="2024 = saison 2024/25")
    ap.add_argument("--phase", default="TOURNAMENT", help="TOURNAMENT / QUALIFICATION …")
    ap.add_argument("--from-matches", default=str(csv_path_raw(MATCHES_FILE)))
    ap.add_argument("--teamstats-out", default=str(csv_path_raw("match_stats.csv")))
    ap.add_argument("--players-pm-out", default=str(csv_path_raw("players_stats_per_match.csv")))
    ap.add_argument("--players-agg-out", default=str(csv_path_raw(PLAYERS_FILE)))

    args = ap.parse_args()

    if args.what == "matches":
        scrape_matches(limit_per_page=args.limit, verbose=True)

    elif args.what == "groups":
        scrape_groups(verbose=True)

    elif args.what == "teamstats":
        scrape_team_stats(from_matches_csv=args.from_matches, out_csv=args.teamstats_out)

    elif args.what == "players_pm":
        scrape_player_stats(from_matches_csv=args.from_matches,
                            out_csv=args.players_pm_out,
                            aggregate_out=args.players_agg_out)

    elif args.what == "players_fast":
        scrape_players_compstats_fast(season_year=args.season_year, phase=args.phase)

    elif args.what == "all":
        scrape_all(limit_per_page=args.limit, season_year=args.season_year, phase=args.phase)
