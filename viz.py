"""
UCLVisualizer — composant de visualisation Plotly
=================================================
Ce module centralise toutes les figures nécessaires pour analyser
la Ligue des champions : distributions, timelines, standings, heatmaps, radars,
comparaisons d'équipes/joueurs, et un "bracket" style Sankey.

Objectifs de design :
- Robuste aux variations de noms de colonnes (normalisation en amont).
- Graphiques "failsafe" : si une info manque → renvoie un graphique vide
  annoté plutôt que d'échouer.
- Palette cohérente avec la charte UEFA depuis `config`.
- Pas de mutation sur les DataFrames d'entrée (copies systématiques).
- Aucune modification de la logique métier d'origine.

Dépendances : plotly, pandas, numpy (optionnel : statsmodels pour la régression).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Iterable
from config import UEFA_BLUE, UEFA_LIGHT_BLUE, GOLD, WHITE, LIGHT_GRAY, PLOTLY_TEMPLATE

# Détection facultative de statsmodels (pour tracer une regression OLS si dispo)
try:
    import statsmodels  # noqa: F401
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


class UCLVisualizer:
    """
    Regroupe tous les graphiques Plotly. La palette est figée lors de l'init
    pour garantir l'homogénéité visuelle.
    """
    def __init__(self):
        # Palette "UEFA-like" + couleurs utilitaires
        self.color_palette = {
            "primary": UEFA_BLUE,
            "secondary": UEFA_LIGHT_BLUE,
            "accent": GOLD,
            "background": WHITE,
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "muted": "#8A8D91",
        }
        # Cycle utilisé pour des séries multiples (radars, scatters multi-équipes)
        self._cycle = [
            self.color_palette["primary"],
            self.color_palette["secondary"],
            self.color_palette["accent"],
            self.color_palette["success"],
            self.color_palette["warning"],
            self.color_palette["danger"],
            self.color_palette["muted"],
        ]

    # ---------- utilitaires ----------
    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise quelques noms usuels et dérive `total_goals` si possible,
        sans toucher à l'objet d'origine (copie côté appelant).
        """
        if df is None or df.empty:
            return df
        # Mapping "lower() → nom réel" pour reconnaître les colonnes
        cols = {c.lower(): c for c in df.columns}

        def get(*names):
            for n in names:
                if n.lower() in cols:
                    return cols[n.lower()]
            return None

        # date_utc / date
        date_col = get("date_utc", "date")
        if date_col and date_col != "date_utc":
            df = df.rename(columns={date_col: "date_utc"})

        # team
        team_col = get("team", "team_name", "club")
        if team_col and team_col != "team":
            df = df.rename(columns={team_col: "team"})

        # player_name
        pcol = get("player_name", "name", "player")
        if pcol and pcol != "player_name":
            df = df.rename(columns={pcol: "player_name"})

        # home/away goals (tolère score_* ou *__score)
        h = get("home_goals", "home_score", "home")
        a = get("away_goals", "away_score", "away")
        if h and h != "home_goals":
            df = df.rename(columns={h: "home_goals"})
        if a and a != "away_goals":
            df = df.rename(columns={a: "away_goals"})

        # total_goals si les deux colonnes existent
        if "total_goals" not in df.columns and {"home_goals", "away_goals"} <= set(df.columns):
            df["total_goals"] = (
                pd.to_numeric(df["home_goals"], errors="coerce").fillna(0)
                + pd.to_numeric(df["away_goals"], errors="coerce").fillna(0)
            )
        return df

    @staticmethod
    def _empty_chart(message: str):
        """
        Fabrique un "placeholder" propre quand les données sont insuffisantes.
        Permet à l'app de ne jamais planter côté UI.
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message, xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            font=dict(size=16, color=UEFA_BLUE)
        )
        fig.update_layout(
            template=None,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=380
        )
        return fig

    # ---------- Matchs ----------
    def create_goals_distribution(self, matches_df: pd.DataFrame):
        """Histogramme des buts totaux par match."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if "total_goals" not in df.columns:
            return self._empty_chart("Missing columns to compute total goals")
        fig = px.histogram(
            df, x="total_goals", nbins=10, title="Goals Distribution in Matches",
            labels={"total_goals": "Total Goals per Match", "count": "Number of Matches"},
            color_discrete_sequence=[self.color_palette["primary"]],
        )
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
            title_font_size=20, bargap=0.05, showlegend=False
        )
        return fig

    def create_match_timeline(self, matches_df: pd.DataFrame):
        """Courbe du nombre de matchs par jour (timeline)."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match timeline data available")
        df = self._ensure_columns(matches_df.copy())
        if "date_utc" not in df.columns:
            return self._empty_chart("No date column (date/date_utc)")
        try:
            df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce").dt.date
        except Exception:
            return self._empty_chart("Invalid date format")
        daily = df.dropna(subset=["date_utc"]).groupby("date_utc").size().reset_index(name="matches_count")
        if daily.empty:
            return self._empty_chart("No valid dates to plot")
        fig = px.line(
            daily, x="date_utc", y="matches_count", markers=True,
            title="Matches Timeline",
            labels={"date_utc": "Date", "matches_count": "Number of Matches"},
        )
        fig.update_traces(line_color=self.color_palette["primary"])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), title_font_size=20)
        return fig

    def create_result_breakdown(self, matches_df: pd.DataFrame):
        """Répartition H/D/A sous forme de pie chart."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if "result" not in df.columns:
            return self._empty_chart("Missing result column")
        counts = df["result"].value_counts(dropna=True).rename({"H": "Home Win", "A": "Away Win", "D": "Draw"})
        out = counts.reindex(["Home Win", "Draw", "Away Win"]).fillna(0)
        fig = px.pie(values=out.values, names=out.index, title="Result Breakdown (H/D/A)", hole=0.35)
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE))
        return fig

    def create_home_away_goals_box(self, matches_df: pd.DataFrame):
        """Boxplot distribution des buts à domicile vs extérieur."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if not {"home_goals", "away_goals"}.issubset(df.columns):
            return self._empty_chart("Missing goals columns")
        melted = pd.melt(
            df[["home_goals", "away_goals"]].rename(columns={"home_goals": "Home", "away_goals": "Away"}),
            value_vars=["Home", "Away"], var_name="Side", value_name="Goals",
        )
        fig = px.box(melted, x="Side", y="Goals", title="Home vs Away Goals Distribution")
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE))
        return fig

    def create_attendance_top_stadiums(self, matches_df: pd.DataFrame, top_n: int = 10):
        """Top stades par affluence moyenne."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if {"stadium", "attendance"} - set(df.columns):
            return self._empty_chart("Missing attendance/stadium columns")
        agg = (
            df.dropna(subset=["stadium"])
              .assign(attendance=pd.to_numeric(df["attendance"], errors="coerce"))
              .groupby("stadium")["attendance"].mean().nlargest(top_n).reset_index()
        )
        if agg.empty:
            return self._empty_chart("No attendance data")
        fig = px.bar(
            agg, x="attendance", y="stadium", orientation="h",
            title="Top Stadiums by Average Attendance",
            labels={"attendance": "Avg Attendance", "stadium": "Stadium"},
            color="attendance", color_continuous_scale=["#d9ebff", UEFA_BLUE],
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), coloraxis_showscale=False)
        return fig

    def create_referees_appearances(self, matches_df: pd.DataFrame, top_n: int = 10):
        """Arbitres les plus désignés (nombre de matchs)."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if "referee" not in df.columns:
            return self._empty_chart("Missing referee column")
        c = df["referee"].dropna().value_counts().nlargest(top_n).reset_index()
        if c.empty:
            return self._empty_chart("No referee data")
        c.columns = ["referee", "matches"]
        fig = px.bar(
            c, x="matches", y="referee", orientation="h",
            title="Top Referees by Matches Officiated",
            labels={"matches": "Matches", "referee": "Referee"},
            color="matches", color_continuous_scale=["#d0e0ff", UEFA_BLUE],
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), coloraxis_showscale=False)
        return fig

    def create_results_heatmap(self, matches_df: pd.DataFrame, min_matches: int = 3):
        """
        Heatmap "points moyens vs adversaire" (vue par équipe).
        On double chaque match (perspective domicile ET extérieur).
        """
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        need = {"home_team", "away_team", "home_goals", "away_goals"}
        if not need.issubset(df.columns):
            return self._empty_chart("Missing columns for heatmap")
        # Points domicile/extérieur
        df["home_pts"] = np.where(
            df["home_goals"] > df["away_goals"], 3,
            np.where(df["home_goals"] == df["away_goals"], 1, 0)
        )
        df["away_pts"] = np.where(
            df["away_goals"] > df["home_goals"], 3,
            np.where(df["away_goals"] == df["home_goals"], 1, 0)
        )
        rows = []
        for _, r in df.iterrows():
            rows.append({"team": r["home_team"], "opponent": r["away_team"], "points": r["home_pts"]})
            rows.append({"team": r["away_team"], "opponent": r["home_team"], "points": r["away_pts"]})
        mat = pd.DataFrame(rows)
        # Option : on filtre les équipes trop peu vues pour éviter une matrice vide
        counts = mat.groupby("team").size()
        keep = counts[counts >= min_matches].index
        mat = mat[mat["team"].isin(keep)]
        pivot = mat.pivot_table(index="team", columns="opponent", values="points", aggfunc="mean").fillna(0)
        if pivot.empty:
            return self._empty_chart("Insufficient matches for heatmap")
        fig = px.imshow(pivot, aspect="auto", title="Results Heatmap (Avg Points vs Opponent)")
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), height=650)
        return fig

    def create_over_under_chart(self, matches_df: pd.DataFrame, line: float = 2.5):
        """Comptage des matchs au-dessus/en-dessous d’un seuil de buts."""
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if "total_goals" not in df.columns:
            return self._empty_chart("Missing total_goals")
        df["bucket"] = np.where(df["total_goals"] > line, f"Over {line}", f"Under/Equal {line}")
        c = df["bucket"].value_counts().rename_axis("bucket").reset_index(name="matches")
        fig = px.bar(c, x="bucket", y="matches", title=f"Over/Under {line} goals")
        fig.update_traces(marker_color=[self.color_palette["secondary"], self.color_palette["primary"]])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE))
        return fig

    # ---------- Équipes / Groupes ----------
    def _normalize_standings(self, standings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise la table de standings (alias de colonnes fréquents)
        et complète Pld/Pts/GD si calculables.
        """
        if standings_df is None or standings_df.empty:
            return standings_df.copy()
        df = standings_df.copy()
        alias_map = {
            "group": {"group", "grp", "group_name", "group_label", "pool"},
            "team": {"team", "team_name", "club"},
            "played": {"played", "matches_played", "mp", "pld", "gp", "games"},
            "won": {"won", "w", "wins", "victories"},
            "drawn": {"drawn", "d", "draws", "ties", "t"},
            "lost": {"lost", "l", "losses", "defeats"},
            "goals_for": {"goals_for", "gf", "for", "goals_scored"},
            "goals_against": {"goals_against", "ga", "against", "goals_conceded"},
            "points": {"points", "pts"},
            "goal_difference": {"goal_difference", "gd", "diff"},
        }
        lower_to_actual = {c.lower(): c for c in df.columns}
        for std, aliases in alias_map.items():
            for a in aliases:
                if a in lower_to_actual and lower_to_actual[a] != std and std not in df.columns:
                    df = df.rename(columns={lower_to_actual[a]: std})
                    break

        # Conversions numériques
        for c in ["played", "won", "drawn", "lost", "goals_for", "goals_against", "points", "goal_difference"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Calculs dérivés si manquants
        if "played" not in df.columns and {"won", "drawn", "lost"} <= set(df.columns):
            df["played"] = df["won"].fillna(0) + df["drawn"].fillna(0) + df["lost"].fillna(0)
        if "points" not in df.columns and {"won", "drawn"} <= set(df.columns):
            df["points"] = 3 * df["won"].fillna(0) + df["drawn"].fillna(0)
        if "goal_difference" not in df.columns and {"goals_for", "goals_against"} <= set(df.columns):
            df["goal_difference"] = df["goals_for"].fillna(0) - df["goals_against"].fillna(0)
        return df

    def create_group_standings_table(self, standings_df: pd.DataFrame):
        """Tableau formaté des standings par groupe (go.Table)."""
        if standings_df is None or standings_df.empty:
            return self._empty_chart("No standings data available")
        df = self._normalize_standings(standings_df)
        needed = {"group", "team", "played", "won", "drawn", "lost", "goals_for", "goals_against", "points"}
        if not needed.issubset(df.columns):
            return self._empty_chart("Missing columns for standings table")
        df = df.sort_values(by=["group", "points", "goal_difference", "goals_for"],
                            ascending=[True, False, False, False], kind="mergesort")
        header = ["Group", "Team", "Pld", "W", "D", "L", "GF", "GA", "Pts"]
        values = [
            df["group"].astype(str).tolist(),
            df["team"].astype(str).tolist(),
            df["played"].astype("Int64").astype(str).tolist(),
            df["won"].astype("Int64").astype(str).tolist(),
            df["drawn"].astype("Int64").astype(str).tolist(),
            df["lost"].astype("Int64").astype(str).tolist(),
            df["goals_for"].astype("Int64").astype(str).tolist(),
            df["goals_against"].astype("Int64").astype(str).tolist(),
            df["points"].astype("Int64").astype(str).tolist(),
        ]
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header, fill_color=UEFA_BLUE,
                font=dict(color="#EAF2FF", size=12, family="Poppins, sans-serif"),
                align="center", height=32,
            ),
            cells=dict(
                values=values, fill_color="rgba(255,255,255,0.03)",
                font=dict(color="#0c2340", size=11, family="Poppins, sans-serif"),
                align="center", height=30,
            ),
            columnwidth=[70, 190, 50, 40, 40, 40, 50, 50, 50],
        )])
        fig.update_layout(title="Group Standings", paper_bgcolor="rgba(0,0,0,0)",
                          height=max(420, 30 * len(df) + 90), margin=dict(l=0, r=0, t=40, b=0))
        return fig

    def create_group_points_gd_scatter(self, standings_df: pd.DataFrame):
        """Dispersion Points vs Différence de buts (taille = buts marqués), coloré par groupe."""
        if standings_df is None or standings_df.empty:
            return self._empty_chart("No standings data available")
        df = self._normalize_standings(standings_df)
        needed = {"team", "group", "points", "goal_difference", "goals_for"}
        if not needed.issubset(df.columns):
            return self._empty_chart("Missing columns for scatter")
        fig = px.scatter(
            df, x="points", y="goal_difference", size="goals_for", color="group",
            hover_data=["team", "played", "won", "drawn", "lost", "goals_for", "goals_against"],
            text="team", title="Group Analysis — Points vs Goal Difference (size = GF)",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title="Points", yaxis_title="Goal Difference",
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE)
        )
        return fig

    def create_group_wdl_stacked(self, standings_df: pd.DataFrame, top_per_group: int = 4):
        """Barres empilées W/D/L par équipe, facettées par groupe (top N équipes par groupe)."""
        if standings_df is None or standings_df.empty:
            return self._empty_chart("No standings data available")
        df = self._normalize_standings(standings_df)
        needed = {"team", "group", "won", "drawn", "lost", "points"}
        if not needed.issubset(df.columns):
            return self._empty_chart("Missing columns for W/D/L stacked")
        df = df.sort_values(
            ["group", "points", "goal_difference", "goals_for"],
            ascending=[True, False, False, False]
        )
        if top_per_group is not None:
            df = df.groupby("group", group_keys=False).head(top_per_group)
        long = df.melt(
            id_vars=["team", "group"], value_vars=["won", "drawn", "lost"],
            var_name="result", value_name="count",
        )
        long["result"] = long["result"].map({"won": "W", "drawn": "D", "lost": "L"})
        fig = px.bar(
            long, x="count", y="team", color="result", orientation="h",
            facet_col="group", facet_col_wrap=2, category_orders={"result": ["L", "D", "W"]},
            title=f"Group Analysis — Wins / Draws / Losses (Top {top_per_group} per group)",
            labels={"count": "Matches", "team": "Team", "result": "Result"},
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
                          legend_traceorder="reversed")
        return fig

    # ---------- Comparaisons équipes ----------
    def create_team_comparison(self, teams_df: pd.DataFrame):
        """Comparaison simple des buts pour/contre par équipe (barres groupées)."""
        if teams_df is None or teams_df.empty:
            return self._empty_chart("No team data available")
        df = self._ensure_columns(teams_df.copy())
        need = {"team", "goals_for", "goals_against"}
        if not need.issubset(df.columns):
            return self._empty_chart("Required columns missing: team, goals_for, goals_against")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Goals For", x=df["team"], y=df["goals_for"],
                             marker_color=self.color_palette["primary"]))
        fig.add_trace(go.Bar(name="Goals Against", x=df["team"], y=df["goals_against"],
                             marker_color=self.color_palette["danger"]))
        fig.update_layout(
            title="Team Goals Comparison", xaxis_title="Team", yaxis_title="Goals",
            barmode="group", plot_bgcolor=WHITE, paper_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
        )
        return fig

    def create_team_radar_chart(self, teams_df: pd.DataFrame, top_n: int = 3):
        """
        Radar multi-équipes (normalisé [0,1]) sur les métriques disponibles :
        goals_for / goals_against / shots / xg / possession (si présents).
        """
        if teams_df is None or teams_df.empty:
            return self._empty_chart("No team data available")
        df = self._ensure_columns(teams_df.copy())
        metrics = []
        if "goals_for" in df.columns: metrics.append("goals_for")
        if "goals_against" in df.columns: metrics.append("goals_against")
        if "shots" in df.columns: metrics.append("shots")
        if "xg" in df.columns: metrics.append("xg")
        if "possession" in df.columns: metrics.append("possession")
        if not metrics:
            return self._empty_chart("No comparable metrics found for teams")

        # Sélection des équipes (top_n sur goals_for si dispo, sinon sur somme des métriques)
        if "goals_for" in df.columns:
            top_teams = df.sort_values("goals_for", ascending=False).head(top_n)
        else:
            num = df[metrics].select_dtypes(include=[np.number])
            top_teams = df.assign(_score=num.sum(axis=1)).sort_values("_score", ascending=False).head(top_n)

        # Normalisation min-max (penalise GA en négatif pour inverser l’échelle)
        def minmax(s: pd.Series):
            s = pd.to_numeric(s, errors="coerce")
            mn, mx = s.min(), s.max()
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                return pd.Series([0.5] * len(s), index=s.index)
            return (s - mn) / (mx - mn)

        norm_df = top_teams.copy()
        for m in metrics:
            col = pd.to_numeric(norm_df[m], errors="coerce")
            if m == "goals_against":
                col = -col
            norm_df[m + "_n"] = minmax(col)

        radar_metrics = [m + "_n" for m in metrics]
        categories = [m.replace("_", " ").title() for m in metrics]
        fig = go.Figure()
        for j, (_, row) in enumerate(norm_df.iterrows()):
            color = self._cycle[j % len(self._cycle)]
            fig.add_trace(go.Scatterpolar(
                r=row[radar_metrics].values.tolist(),
                theta=categories, fill="toself",
                name=row.get("team", f"Team {j+1}"),
                line=dict(color=color), fillcolor=color, opacity=0.35,
            ))
        fig.update_layout(
            title="Team Radar (normalized)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
        )
        return fig

    def create_team_style_scatter(self, teams_df: pd.DataFrame):
        """
        Carte de style d’équipe : axe X = attaque (buts pour),
        axe Y = "solidité défensive" (−xGA ou −GA), taille = tirs si dispo.
        """
        if teams_df is None or teams_df.empty:
            return self._empty_chart("No team data available")
        df = self._ensure_columns(teams_df.copy())
        if "team" not in df.columns or "goals_for" not in df.columns:
            return self._empty_chart("Need at least team and goals_for")
        if "xga" in df.columns:
            df["defense_index"] = -pd.to_numeric(df["xga"], errors="coerce")
        elif "goals_against" in df.columns:
            df["defense_index"] = -pd.to_numeric(df["goals_against"], errors="coerce")
        else:
            df["defense_index"] = 0
        size_col = "shots" if "shots" in df.columns else None
        fig = px.scatter(
            df, x="goals_for", y="defense_index", text="team", size=size_col,
            hover_data=df.columns, title="Team Style Map (Attack vs Defensive Solidity)"
        )
        fig.update_traces(marker=dict(color=self.color_palette["primary"], line=dict(width=0)))
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
            xaxis_title="Goals For (higher = more attacking)",
            yaxis_title="Defensive Solidity (higher = better)",
        )
        return fig

    # ---------- Progression / Histoires ----------
    def create_team_progression(self, matches_df: pd.DataFrame, team: str):
        """Cumul des points dans le temps pour une équipe donnée."""
        if matches_df is None or matches_df.empty or not team:
            return self._empty_chart("No data/team for progression")
        df = self._ensure_columns(matches_df.copy())
        need = {"date_utc", "home_team", "away_team", "home_goals", "away_goals"}
        if not need.issubset(df.columns):
            return self._empty_chart("Missing columns for progression")
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
        df = df.dropna(subset=["date_utc"]).sort_values("date_utc")

        def pts(row):
            if row["home_team"] == team:
                if row["home_goals"] > row["away_goals"]:
                    return 3
                if row["home_goals"] == row["away_goals"]:
                    return 1
                return 0
            if row["away_team"] == team:
                if row["away_goals"] > row["home_goals"]:
                    return 3
                if row["away_goals"] == row["home_goals"]:
                    return 1
                return 0
            return np.nan

        df["pts"] = df.apply(pts, axis=1)
        sub = df.dropna(subset=["pts"])[["date_utc", "pts"]].copy()
        if sub.empty:
            return self._empty_chart(f"No matches found for {team}")
        sub["cum_points"] = sub["pts"].cumsum()
        fig = px.line(sub, x="date_utc", y="cum_points", title=f"{team} — Cumulative Points")
        fig.update_traces(line_color=self.color_palette["primary"])
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
            xaxis_title="Date", yaxis_title="Points (cumulative)",
        )
        return fig

    def create_team_momentum(self, matches_df: pd.DataFrame, team: str, window: int = 3):
        """Momentum = moyenne glissante des points (fenêtre configurable)."""
        if matches_df is None or matches_df.empty or not team:
            return self._empty_chart("No data/team for momentum")
        df = self._ensure_columns(matches_df.copy())
        need = {"date_utc", "home_team", "away_team", "home_goals", "away_goals"}
        if not need.issubset(df.columns):
            return self._empty_chart("Missing columns for momentum")
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
        df = df.dropna(subset=["date_utc"]).sort_values("date_utc")

        def pts(row):
            if row["home_team"] == team:
                return 3 if row["home_goals"] > row["away_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0)
            if row["away_team"] == team:
                return 3 if row["away_goals"] > row["home_goals"] else (1 if row["away_goals"] == row["home_goals"] else 0)
            return np.nan

        sub = df.assign(pts=df.apply(pts, axis=1)).dropna(subset=["pts"])[["date_utc", "pts"]]
        if sub.empty:
            return self._empty_chart(f"No matches found for {team}")
        sub["rolling_pts"] = sub["pts"].rolling(window, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub["date_utc"], y=sub["pts"], mode="markers+lines",
            name="Points", line=dict(color=self.color_palette["secondary"], width=1)
        ))
        fig.add_trace(go.Scatter(
            x=sub["date_utc"], y=sub["rolling_pts"], mode="lines",
            name=f"Rolling ({window})", line=dict(color=self.color_palette["primary"], width=3)
        ))
        fig.update_layout(
            title=f"{team} — Momentum (rolling {window})",
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
            xaxis_title="Date", yaxis_title="Points per match",
        )
        return fig

    # ---------- Sunburst Phase -> Team (robuste, sans px.sunburst) ----------
    def create_phase_sunburst(self, matches_df: pd.DataFrame):
        """
        Sunburst "Phase → Équipe" pondéré par les buts marqués.
        Normalise grossièrement la phase (group/round/quarter/semi/final).
        """
        if matches_df is None or matches_df.empty:
            return self._empty_chart("No match data available")
        df = self._ensure_columns(matches_df.copy())
        if not {"home_team", "away_team", "home_goals", "away_goals", "stage"}.issubset(df.columns):
            return self._empty_chart("Missing columns for sunburst")

        def norm_phase(s: str) -> str:
            s = (s or "").lower()
            if "group" in s: return "Group Stage"
            if "round" in s: return "Round of 16"
            if "quarter" in s: return "Quarter-finals"
            if "semi" in s: return "Semi-finals"
            if "final" in s and "semi" not in s: return "Final"
            return s.title() if s else "Unknown"

        df["phase"] = df["stage"].map(norm_phase)

        rows = []
        for _, r in df.iterrows():
            rows.append({"phase": r["phase"], "team": r["home_team"], "goals": pd.to_numeric(r["home_goals"], errors="coerce")})
            rows.append({"phase": r["phase"], "team": r["away_team"], "goals": pd.to_numeric(r["away_goals"], errors="coerce")})
        long = pd.DataFrame(rows).dropna()
        agg = long.groupby(["phase", "team"], as_index=False)["goals"].sum()
        if agg.empty:
            return self._empty_chart("No goals data")

        phase_tot = agg.groupby("phase", as_index=False)["goals"].sum()

        labels = ["Goals"]; parents = [""]; values = [float(phase_tot["goals"].sum())]
        for _, r in phase_tot.iterrows():
            labels.append(r["phase"]); parents.append("Goals"); values.append(float(r["goals"]))
        for _, r in agg.iterrows():
            labels.append(r["team"]); parents.append(r["phase"]); values.append(float(r["goals"]))

        fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total"))
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
                          title="Goals by Phase → Team (Sunburst)")
        return fig

    # ---------- Joueurs ----------
    def create_top_scorers_bar(self, players_df: pd.DataFrame, top_n: int = 15):
        """Top buteurs (barres horizontales)."""
        if players_df is None or players_df.empty:
            return self._empty_chart("No player data available")
        df = self._ensure_columns(players_df.copy())
        if "goals" not in df.columns:
            gcol = [c for c in df.columns if c.lower() in ("but", "buts", "goals_scored", "goals")]
            if gcol: df = df.rename(columns={gcol[0]: "goals"})
        if "team" not in df.columns:
            tcol = [c for c in df.columns if c.lower() in ("club", "equipe", "team_name")]
            if tcol: df = df.rename(columns={tcol[0]: "team"})
        if not {"player_name", "goals"}.issubset(df.columns):
            return self._empty_chart("Missing player_name/goals")
        top = df.sort_values("goals", ascending=False).head(top_n)
        if top.empty:
            return self._empty_chart("No goals data")
        fig = px.bar(
            top, x="goals", y="player_name", orientation="h",
            color="goals", color_continuous_scale=["#d0e0ff", UEFA_BLUE],
            hover_data=["team"] if "team" in top.columns else None,
            title=f"Top {len(top)} Scorers",
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), coloraxis_showscale=False)
        return fig

    def create_top_assists_bar(self, players_df: pd.DataFrame, top_n: int = 15):
        """Top passeurs décisifs (barres horizontales)."""
        if players_df is None or players_df.empty:
            return self._empty_chart("No player data available")
        df = self._ensure_columns(players_df.copy())
        if "assists" not in df.columns:
            acol = [c for c in df.columns if c.lower() in ("passes_decisives", "assists", "assist")]
            if acol: df = df.rename(columns={acol[0]: "assists"})
        if "team" not in df.columns:
            tcol = [c for c in df.columns if c.lower() in ("club", "equipe", "team_name")]
            if tcol: df = df.rename(columns={tcol[0]: "team"})
        if not {"player_name", "assists"}.issubset(df.columns):
            return self._empty_chart("Missing player_name/assists")
        top = df.sort_values("assists", ascending=False).head(top_n)
        if top.empty:
            return self._empty_chart("No assists data")
        fig = px.bar(
            top, x="assists", y="player_name", orientation="h",
            color="assists", color_continuous_scale=["#ffe3a3", GOLD],
            hover_data=["team"] if "team" in top.columns else None,
            title=f"Top {len(top)} Assist Providers",
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), coloraxis_showscale=False)
        return fig

    def create_goal_involvement(self, players_df: pd.DataFrame, top_n: int = 15):
        """
        Classement par implication décisive (G + A).
        La métrique `GI` est robuste aux NaN (coerce + fillna).
        """
        if players_df is None or players_df.empty:
            return self._empty_chart("No player data available")
        df = self._ensure_columns(players_df.copy())
        for col in ["goals", "assists"]:
            if col not in df.columns:
                df[col] = 0
        df["GI"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0) + \
                   pd.to_numeric(df["assists"], errors="coerce").fillna(0)
        if "team" not in df.columns and "team_name" in df.columns:
            df = df.rename(columns={"team_name": "team"})
        top = df.sort_values("GI", ascending=False).head(top_n)
        fig = px.bar(
            top, x="GI", y="player_name", orientation="h",
            color="GI", color_continuous_scale=["#eaf7ff", UEFA_LIGHT_BLUE],
            hover_data=["goals", "assists", "team"] if "team" in top.columns else ["goals", "assists"],
            title=f"Top {len(top)} — Goal Involvements (G + A)",
        )
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE), coloraxis_showscale=False)
        return fig

    def create_player_radar(self, players_df: pd.DataFrame, player_name: str):
        """
        Radar individuel pour un joueur donné.
        Métriques supportées si présentes : goals, assists, shots, key_passes,
        xg, xa, dribbles_won, tackles (normalisation min-max).
        """
        if players_df is None or players_df.empty or not player_name:
            return self._empty_chart("No player / data for radar")
        df = self._ensure_columns(players_df.copy())
        # Rattrapage d'alias fréquents
        aliases = {
            "key_passes": {"key_passes", "passes_clés", "keypasses"},
            "dribbles_won": {"dribbles_won", "dribbles_reussis", "dribbles_successful"},
        }
        for std, al in aliases.items():
            if std not in df.columns:
                cand = [c for c in df.columns if c.lower() in al]
                if cand:
                    df = df.rename(columns={cand[0]: std})
        metrics_pref = ["goals", "assists", "shots", "key_passes", "xg", "xa", "dribbles_won", "tackles"]
        metrics = [m for m in metrics_pref if m in df.columns]
        if not metrics:
            return self._empty_chart("No comparable player metrics found")

        # Normalisation min-max (0.5 si plage dégénérée)
        def minmax(s: pd.Series):
            s = pd.to_numeric(s, errors="coerce")
            mn, mx = s.min(), s.max()
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                return pd.Series([0.5] * len(s), index=s.index)
            return (s - mn) / (mx - mn)

        sub = df.copy()
        for m in metrics:
            sub[m + "_n"] = minmax(sub[m])

        target = sub[sub["player_name"].astype(str).str.lower() == str(player_name).lower()]
        if target.empty:
            return self._empty_chart(f"Player not found: {player_name}")
        r_vals = target[[m + "_n" for m in metrics]].iloc[0].tolist()
        categories = [m.replace("_", " ").title() for m in metrics]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=categories, fill="toself", name=player_name,
            line=dict(color=self.color_palette["primary"]),
            fillcolor=self.color_palette["secondary"], opacity=0.4,
        ))
        fig.update_layout(
            title=f"{player_name} — Radar (normalized)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False, paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
        )
        return fig

    def create_age_distribution(self, players_df: pd.DataFrame):
        """Distribution de l'âge des joueurs (fallback via birth_year si fourni)."""
        if players_df is None or players_df.empty:
            return self._empty_chart("No player data available")
        df = self._ensure_columns(players_df.copy())
        if "age" not in df.columns and "birth_year" in df.columns:
            df["age"] = pd.Timestamp.now().year - pd.to_numeric(df["birth_year"], errors="coerce")
        if "age" not in df.columns:
            return self._empty_chart("Missing age/birth_year to compute distribution")
        fig = px.histogram(df, x="age", nbins=20, title="Players Age Distribution")
        fig.update_traces(marker=dict(color=self.color_palette["primary"]))
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
            xaxis_title="Age", yaxis_title="Players",
        )
        return fig

    def create_minutes_vs_goals(self, players_df: pd.DataFrame):
        """
        Nuage de points Minutes vs Buts. Ajoute une droite de régression si
        `statsmodels` est installé, sinon affiche un tip discret.
        """
        if players_df is None or players_df.empty:
            return self._empty_chart("No player data available")
        df = self._ensure_columns(players_df.copy())
        if "minutes" not in df.columns:
            mcol = [c for c in df.columns if c.lower() in ("minutes", "mins_played", "time_played", "minutes_played")]
            if mcol:
                df = df.rename(columns={mcol[0]: "minutes"})
        if "goals" not in df.columns:
            gcol = [c for c in df.columns if c.lower() in ("goals", "buts")]
            if gcol:
                df = df.rename(columns={gcol[0]: "goals"})
        if not {"player_name", "minutes", "goals"}.issubset(df.columns):
            return self._empty_chart("Need player_name/minutes/goals")
        trend = "ols" if _HAS_STATSMODELS else None
        fig = px.scatter(
            df, x="minutes", y="goals", hover_name="player_name",
            title="Minutes Played vs Goals", trendline=trend,
        )
        fig.update_traces(marker=dict(color=self.color_palette["primary"]))
        if not _HAS_STATSMODELS:
            fig.add_annotation(
                text="Tip: install 'statsmodels' to display the regression line",
                xref="paper", yref="paper", x=0.01, y=1.08, showarrow=False, font=dict(color="#555", size=11),
            )
        fig.update_layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=dict(color=UEFA_BLUE),
            xaxis_title="Minutes", yaxis_title="Goals",
        )
        return fig

    # ---------- Bracket ----------
    def create_tournament_bracket(self, bracket_data: dict):
        """
        Représentation "flow" des qualifs au format Sankey.
        `bracket_data` doit contenir des listes par clé :
          - round_of_16, quarter_finals, semi_finals, final
        Chaque match : {home, away, home_score, away_score, winner?}
        Si winner absent, il est déduit du score.
        """
        if not bracket_data:
            return self._empty_chart("No bracket data")
        nodes = []

        def idx(name):
            if name not in nodes:
                nodes.append(name)
            return nodes.index(name)

        links = {"source": [], "target": [], "value": [], "label": []}

        def winner(m):
            return m.get("winner") or (m["home"] if m["home_score"] > m["away_score"] else m["away"])

        for rnd_key, lbl in [
            ("round_of_16", "R16"),
            ("quarter_finals", "QF"),
            ("semi_finals", "SF"),
            ("final", "Final"),
        ]:
            for m in bracket_data.get(rnd_key, []):
                w = winner(m)
                links["source"] += [idx(m["home"]), idx(m["away"])]
                links["target"] += [idx(w), idx(w)]
                links["value"] += [1, 1]  # poids unitaire par "chemin"
                links["label"] += [
                    f"{lbl} {m['home_score']}-{m['away_score']}",
                    f"{lbl} {m['home_score']}-{m['away_score']}",
                ]

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=20, thickness=16, color=UEFA_LIGHT_BLUE),
            link=dict(source=links["source"], target=links["target"],
                      value=links["value"], label=links["label"])
        )])
        fig.update_layout(title_text="Tournament Bracket (Flow)", font=dict(color=UEFA_BLUE), paper_bgcolor=WHITE)
        return fig
