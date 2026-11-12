"""Application Streamlit : assistant de selection de portefeuille."""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, prepare_data, profiles
from src.agents import agents

# ---------------------------------------------------------------- libelles
RISKS = {
    "Prudent": "Je veux limiter les pertes, quitte à gagner moins.",
    "Équilibré": "J'accepte des hauts et des bas pour viser un meilleur rendement.",
    "Dynamique": "Je vise un rendement élevé et j'accepte des baisses importantes.",
}
RISK_KEYS = {"Prudent": "Prudent", "Équilibré": "Equilibre", "Dynamique": "Dynamique"}
RISK_LABELS = {v: k for k, v in RISK_KEYS.items()}
OBJECTIVES = {
    "Préserver mon capital": "Preserver",
    "Faire fructifier mon capital": "Fructifier",
    "Maximiser le rendement à long terme": "Maximiser",
}
OBJECTIVE_LABELS = {v: k for k, v in OBJECTIVES.items()}
HORIZONS = ["Moins d'un an", "1 à 3 ans", "3 à 5 ans", "Plus de 5 ans"]
REACTIONS = [
    "Je vendrais pour éviter de perdre davantage",
    "J'attendrais que les marchés se rétablissent",
    "J'en profiterais pour investir davantage",
]
MARKETS = ["Marché marocain", "Marché international", "Sans préférence"]
MODES = {
    "Moyenne des répartitions validées (recommandé)": "consensus",
    "Meilleure répartition seule": "best",
}
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre"]

CLASS_LABELS = {
    "Actions US": "Actions américaines",
    "Actions internationales": "Actions internationales",
    "Obligations": "Obligations",
    "Refuges et diversifiants": "Or, immobilier et matières premières",
    "Liquidités": "Liquidités",
}
CLASS_COLORS = {
    "Actions US": "#0B1F3A",
    "Actions internationales": "#3B6EA5",
    "Obligations": "#0E7C86",
    "Refuges et diversifiants": "#D9A441",
    "Liquidités": "#B8C0CC",
}
FUND_LABELS = {
    "SPY": "Grandes entreprises américaines",
    "QQQ": "Technologie américaine",
    "IWM": "Petites entreprises américaines",
    "EFA": "Pays développés hors États-Unis",
    "EEM": "Pays émergents",
    "AGG": "Obligations américaines diversifiées",
    "TLT": "Obligations d'État à long terme",
    "LQD": "Obligations d'entreprises",
    "GLD": "Or",
    "VNQ": "Immobilier coté",
    "DBC": "Matières premières",
}
APPROACH_LABELS = {
    "1/N": "Répartition égale",
    "Vol inverse": "Priorité aux fonds les plus stables",
    "Min variance": "Variations les plus faibles possibles",
    "Max Sharpe": "Meilleur rapport gain et risque",
    "HRP": "Selon les groupes de fonds",
    "Consensus": "Moyenne des répartitions validées",
}
APPROACH_SENTENCES = {
    "1/N": "une répartition égale entre les fonds",
    "Vol inverse": "la répartition qui donne plus de poids aux fonds les plus stables",
    "Min variance": "la répartition qui limite le plus les variations",
    "Max Sharpe": "la répartition qui offre le meilleur rapport entre gain et risque",
    "HRP": "la répartition construite selon les groupes de fonds qui évoluent ensemble",
}
STRESS = {
    "Crise financière (septembre 2008 à février 2009)": ("2008-09-01", "2009-02-28"),
    "Covid (février et mars 2020)": ("2020-02-01", "2020-03-31"),
    "Hausse des taux (année 2022)": ("2022-01-01", "2022-12-31"),
}
KPI_COLORS = {"gain": "#2E9E6B", "variation": "#3B6EA5", "baisse": "#E0693B", "liquidites": "#D9A441"}
AGENT_COLORS = ["#0E7C86", "#3B6EA5", "#6D5BD0", "#D9A441", "#E0693B", "#2E9E6B", "#0B1F3A"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp h6, .stApp input, .stApp textarea, .stApp button, .stApp th, .stApp td,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-baseweb="select"] div {
  font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
.hero { background: linear-gradient(135deg, #0B1F3A 0%, #12375F 55%, #0E7C86 130%); border-radius: 20px;
  padding: 2.6rem 2.8rem; margin-bottom: 1.4rem; }
.hero h1 { color: #ffffff; font-size: 2.1rem; font-weight: 700; margin: 0 0 .7rem 0; line-height: 1.2;
  letter-spacing: -0.02em; padding: 0; }
.hero p { color: #dbe7f3; font-size: 1.03rem; line-height: 1.65; max-width: 700px; margin: 0; }
.steps { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.step { flex: 1; background: #ffffff; border: 1px solid #E5E9F0; border-top: 4px solid var(--c, #0E7C86);
  border-radius: 14px; padding: 1rem 1.2rem; }
.step-num { color: var(--c, #0E7C86); font-weight: 700; font-size: .8rem; letter-spacing: .08em; }
.step-title { color: #0B1F3A; font-weight: 600; margin: .15rem 0 .25rem 0; }
.step-text { color: #6B7280; font-size: .88rem; line-height: 1.5; }
div[data-testid="stForm"] { background: #ffffff; border: 1px solid #E5E9F0; border-radius: 18px;
  padding: 1.8rem 2rem; box-shadow: 0 1px 3px rgba(16, 24, 40, .05); }
div[data-testid="stFormSubmitButton"] button, div.stButton > button { border-radius: 10px; font-weight: 600;
  min-height: 3rem; }
div[data-testid="stFormSubmitButton"] button { width: 100%; }
.card { background: #ffffff; border: 1px solid #E5E9F0; border-radius: 16px; padding: 1.2rem 1.4rem;
  box-shadow: 0 1px 3px rgba(16, 24, 40, .05); }
.kpi { height: 100%; border-top: 4px solid var(--c, #0E7C86); }
.kpi-label { font-size: .76rem; color: #6B7280; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
.kpi-label::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--c, #0E7C86); margin-right: .5rem; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #0B1F3A; margin: .3rem 0 .35rem 0; letter-spacing: -0.02em; }
.kpi-note { font-size: .85rem; color: #6B7280; line-height: 1.5; }
.page-title { font-size: 1.8rem; font-weight: 700; color: #0B1F3A; letter-spacing: -0.02em; margin: 0; }
.page-sub { color: #6B7280; margin: .3rem 0 1.4rem 0; }
.section-title { font-size: 1.2rem; font-weight: 700; color: #0B1F3A; margin: 2.2rem 0 .9rem 0;
  border-left: 5px solid #0E7C86; padding-left: .8rem; }
.alloc-row { display: flex; align-items: center; gap: .8rem; padding: .7rem 0; border-bottom: 1px solid #EEF1F5; }
.alloc-row:last-child { border-bottom: none; }
.dot { width: 12px; height: 12px; border-radius: 50%; flex: 0 0 12px; }
.alloc-name { flex: 1; color: #1F2937; font-weight: 500; }
.alloc-pct { color: #0B1F3A; font-weight: 700; width: 3.5rem; text-align: right; }
.alloc-amt { color: #6B7280; width: 8rem; text-align: right; font-size: .92rem; }
.points { margin: 0; padding-left: 1.1rem; }
.points li { color: #374151; line-height: 1.7; margin-bottom: .45rem; }
.points li::marker { color: #0E7C86; }
.agent { display: flex; gap: 1rem; padding: .95rem 0; border-bottom: 1px solid #EEF1F5; }
.agent:last-child { border-bottom: none; }
.agent-num { flex: 0 0 2rem; height: 2rem; border-radius: 50%; font-weight: 700; display: flex;
  align-items: center; justify-content: center; }
.agent-name { font-weight: 600; color: #0B1F3A; margin-bottom: .1rem; }
.agent-text { color: #4B5563; font-size: .94rem; line-height: 1.6; }
.legal { color: #9CA3AF; font-size: .8rem; line-height: 1.6; margin-top: 2.5rem; }
</style>
"""


# ---------------------------------------------------------------- utilitaires
def md(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def dh(x: float) -> str:
    return f"{x:,.0f}".replace(",", "\u00a0") + "\u00a0DH"


def fr_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}\u00a0%".replace(".", ",")


def fr_signed(x: float) -> str:
    return f"{x * 100:+.1f}\u00a0%".replace(".", ",")


def fr_month(ts: pd.Timestamp) -> str:
    return f"{MONTHS_FR[ts.month - 1]} {ts.year}"


def parse_capital(text: str):
    """Lit un montant saisi librement : '100000', '100 000', '100 000 DH'."""
    cleaned = re.sub(r"(dh|mad|dirhams?)", "", text.lower())
    cleaned = cleaned.replace("\u202f", "").replace("\u00a0", "").replace(" ", "").replace(",", ".")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


@st.cache_data
def load_returns() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "returns_monthly.csv"
    if not path.exists():           # permet un deploiement a partir des seules donnees brutes
        prepare_data.main()
    return pd.read_csv(path, index_col=0, parse_dates=True)


def color_signed(v) -> str:
    if isinstance(v, str) and v.startswith("-"):
        return "color: #B42318; font-weight: 600"
    if isinstance(v, str) and v.startswith("+"):
        return "color: #067647; font-weight: 600"
    return ""


def color_result(v) -> str:
    return {
        "Retenue": "background-color: #D1FADF; color: #05603A; font-weight: 600",
        "Validée": "background-color: #DBEAFE; color: #1E40AF; font-weight: 600",
        "Écartée": "background-color: #FEE4E2; color: #B42318; font-weight: 600",
    }.get(v, "")


# ---------------------------------------------------------------- calculs d'affichage
def by_class(ctx) -> dict:
    out = {}
    for t, w in zip(ctx.assets, ctx.selected.weights):
        cls = config.UNIVERSE[t][1]
        out[cls] = out.get(cls, 0.0) + float(w)
    out["Liquidités"] = ctx.selected.cash
    return {k: v for k, v in out.items() if v > 0.005}


def reference_weights(ctx) -> np.ndarray:
    mix = profiles.REFERENCE_MIXES[ctx.profile.risk]
    return np.array([mix.get(t, 0.0) for t in ctx.assets])


def humanize(reason: str) -> str:
    if "sous la bande" in reason:
        return "Trop prudente pour votre profil"
    if "au-dessus de la bande" in reason:
        return "Trop risquée pour votre profil"
    if "depasse" in reason:
        return "Trop concentrée sur un seul fonds"
    if "diversification" in reason:
        return "Pas assez diversifiée"
    return reason


def method_text(ctx) -> str:
    s = ctx.selected
    n_ok = sum(c.accepted for c in ctx.candidates)
    if s.name == "Consensus":
        return f"la moyenne des {n_ok} répartitions qui respectaient ces limites"
    return APPROACH_SENTENCES.get(s.name, s.name)


def notices(ctx, answers: dict) -> list:
    p, out = ctx.profile, []
    if answers["market"] == "Marché marocain":
        out.append(("info", "Le marché marocain n'est pas encore couvert. La proposition ci-dessous porte sur "
                            "des fonds internationaux cotés aux États-Unis."))
    if p.horizon == "Moins d'un an" and p.risk == "Dynamique":
        out.append(("warning", "Avec un horizon inférieur à un an, un profil Dynamique est risqué : "
                               "une baisse temporaire pourrait ne pas avoir le temps d'être rattrapée."))
    if p.objective == "Maximiser" and p.risk == "Prudent":
        out.append(("info", "Vous visez un rendement maximal avec un profil Prudent. Nous avons respecté "
                            "la limite de risque de votre profil, ce qui plafonne le rendement possible."))
    if answers["reaction"] == REACTIONS[0] and p.risk in ("Equilibre", "Dynamique"):
        out.append(("warning", "Vous nous dites que vous vendriez en cas de forte baisse. Un profil "
                               f"{RISK_LABELS[p.risk].lower()} peut connaître des baisses importantes "
                               "pendant plusieurs mois. Un profil plus prudent vous conviendrait peut-être mieux."))
    if answers["reaction"] == REACTIONS[2] and p.risk == "Prudent":
        out.append(("info", "Vous seriez à l'aise pour investir davantage après une baisse. "
                            "Un profil plus équilibré pourrait mieux vous correspondre."))
    if not any(c.accepted for c in ctx.candidates):
        out.append(("warning", "Aucune répartition ne respectait toutes les limites de risque. Nous avons gardé "
                               "la plus proche de votre profil, à prendre avec prudence."))
    return out


def takeaways(ctx) -> list:
    s, p = ctx.selected, ctx.profile
    n_funds = int((s.weights > 0.005).sum())
    pts = []
    if s.cash >= 0.30:
        pts.append(f"{fr_pct(s.cash, 0)} de votre capital reste en liquidités. C'est ce qui permet de tenir "
                   "la limite de variation de votre profil.")
    elif s.cash > 0.005:
        pts.append(f"{fr_pct(s.cash, 0)} de votre capital reste en liquidités pour amortir les secousses des marchés.")
    else:
        pts.append("Tout votre capital est investi sur les marchés.")
    pts.append(f"Votre argent est réparti sur {n_funds} fonds différents, pour ne pas dépendre d'un seul marché.")
    if s.worst_month < 0:
        pts.append(f"En année normale, la valeur du portefeuille peut monter ou baisser d'environ "
                   f"{dh(s.vol * p.capital)}. Sur un mois difficile, la baisse a atteint {fr_pct(abs(s.worst_month))} "
                   "ces cinq dernières années.")
    else:
        pts.append(f"En année normale, la valeur du portefeuille peut monter ou baisser d'environ "
                   f"{dh(s.vol * p.capital)}.")
    n_ok = sum(c.accepted for c in ctx.candidates)
    pts.append(f"Nous avons comparé {len(ctx.candidates)} façons de répartir votre capital. {n_ok} respectaient "
               f"vos limites de risque. Nous avons retenu {method_text(ctx)}.")
    return pts


def agent_steps(ctx) -> list:
    p = ctx.profile
    n_rej = sum(not c.accepted for c in ctx.candidates)
    n_groups = len(ctx.analysis["clusters"])
    return [
        ("Agent Données",
         f"Rassemble l'historique mensuel de {len(ctx.assets)} fonds (actions, obligations, or, immobilier, "
         f"matières premières) sur les cinq dernières années, jusqu'en {fr_month(ctx.as_of)}, et mesure les gains "
         "et les risques de chacun."),
        ("Agent Profil",
         f"Traduit vos réponses en une limite de risque : une variation annuelle visée d'environ "
         f"{fr_pct(ctx.target_vol)}."),
        ("Agent Analyse",
         f"Observe comment les fonds évoluent les uns par rapport aux autres et les range en {n_groups} grands "
         "groupes. Quand l'un baisse, d'autres tiennent parfois mieux : c'est la base de la diversification."),
        ("Agent Optimisation",
         f"Construit {len(ctx.candidates)} répartitions possibles, chacune avec sa logique, puis réduit la part "
         "investie si nécessaire pour rester dans votre limite."),
        ("Agent Risque",
         f"Contrôle chaque répartition : niveau de variation, poids sur un même fonds, diversification. "
         f"{n_rej} répartition(s) écartée(s)."),
        ("Agent Sélection",
         f"Choisit parmi les répartitions validées en tenant compte de votre objectif "
         f"({OBJECTIVE_LABELS[p.objective].lower()}). Retenu : {method_text(ctx)}."),
        ("Agent Synthèse", "Rédige la proposition et les explications que vous lisez ici."),
    ]


def approaches_table(ctx) -> pd.DataFrame:
    cands = list(ctx.candidates)
    if not any(c is ctx.selected for c in cands):
        cands.append(ctx.selected)
    rows = []
    for c in cands:
        if c is ctx.selected:
            status = "Retenue"
            why = "; ".join(humanize(r) for r in c.reasons) or "Respecte vos limites"
        elif c.accepted:
            status, why = "Validée", "Respecte vos limites"
        else:
            status, why = "Écartée", "; ".join(humanize(r) for r in c.reasons)
        rows.append({
            "Répartition": APPROACH_LABELS.get(c.name, c.name),
            "Variation annuelle": fr_pct(c.vol),
            "Gain annuel estimé": fr_pct(c.exp_return),
            "Plus forte baisse mensuelle": fr_pct(c.worst_month),
            "Résultat": status,
            "Raison": why,
        })
    return pd.DataFrame(rows)


def stress_table(ctx, rets: pd.DataFrame) -> pd.DataFrame:
    s, ref_w = ctx.selected, reference_weights(ctx)
    rows = []
    for name, (a, b) in STRESS.items():
        r = rets.loc[a:b]
        if r.empty:
            continue
        port = r[ctx.assets].values @ s.weights + s.cash * r["RF"].values
        ref = r[ctx.assets].values @ ref_w
        rows.append({"Épisode": name,
                     "Votre portefeuille": fr_signed((1 + port).prod() - 1),
                     "Mélange classique": fr_signed((1 + ref).prod() - 1),
                     "Actions américaines": fr_signed((1 + r["SPY"]).prod() - 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- graphiques
def donut(ctx) -> go.Figure:
    classes = by_class(ctx)
    fig = go.Figure(go.Pie(
        labels=[CLASS_LABELS[k] for k in classes],
        values=list(classes.values()),
        hole=0.64, sort=False, textinfo="none",
        marker=dict(colors=[CLASS_COLORS[k] for k in classes], line=dict(color="#ffffff", width=3)),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{dh(ctx.profile.capital)}</b>", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=17, color="#0B1F3A"))],
    )
    return fig


def growth_chart(ctx) -> go.Figure:
    hist, s = ctx.hist, ctx.selected
    series = [
        ("Votre portefeuille", hist[ctx.assets].values @ s.weights + s.cash * hist["RF"].values,
         dict(color="#0E7C86", width=3)),
        ("Mélange classique", hist[ctx.assets].values @ reference_weights(ctx),
         dict(color="#9CA3AF", width=2, dash="dash")),
        ("Actions américaines", hist["SPY"].values, dict(color="#0B1F3A", width=1.6)),
    ]
    fig = go.Figure()
    for name, r, line in series:
        fig.add_trace(go.Scatter(x=hist.index, y=100 * np.cumprod(1 + r), name=name, mode="lines", line=line))
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Valeur (base 100)",
        legend=dict(orientation="h", y=1.08, x=0), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#374151"),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#EEF1F5"), hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------- blocs HTML
def kpi(label: str, value: str, note: str, color: str) -> str:
    return (f"<div class='card kpi' style='--c:{color}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>")


def alloc_html(ctx) -> str:
    rows = []
    for key, w in sorted(by_class(ctx).items(), key=lambda kv: -kv[1]):
        rows.append(f"<div class='alloc-row'><span class='dot' style='background:{CLASS_COLORS[key]}'></span>"
                    f"<span class='alloc-name'>{CLASS_LABELS[key]}</span>"
                    f"<span class='alloc-pct'>{fr_pct(w, 0)}</span>"
                    f"<span class='alloc-amt'>{dh(w * ctx.profile.capital)}</span></div>")
    return "<div class='card'>" + "".join(rows) + "</div>"


def agents_html(steps: list) -> str:
    items = []
    for i, (name, text) in enumerate(steps, 1):
        color = AGENT_COLORS[(i - 1) % len(AGENT_COLORS)]
        items.append(f"<div class='agent'><div class='agent-num' style='background:{color}1F;color:{color}'>{i}</div>"
                     f"<div><div class='agent-name'>{name}</div><div class='agent-text'>{text}</div></div></div>")
    return "<div class='card'>" + "".join(items) + "</div>"


# ---------------------------------------------------------------- pages
def form_page(rets: pd.DataFrame) -> None:
    md("<div class='hero'><h1>Un portefeuille construit pour votre situation</h1>"
       "<p>Répondez à quelques questions. Une équipe d'agents d'analyse étudie les marchés, contrôle les risques "
       "et vous propose une répartition de votre capital, avec les raisons de chaque choix.</p></div>")
    md("<div class='steps'>"
       "<div class='step' style='--c:#0E7C86'><div class='step-num'>ÉTAPE 1</div>"
       "<div class='step-title'>Vous décrivez votre situation</div>"
       "<div class='step-text'>Capital, durée, objectif et façon de réagir face aux baisses.</div></div>"
       "<div class='step' style='--c:#3B6EA5'><div class='step-num'>ÉTAPE 2</div>"
       "<div class='step-title'>Les agents analysent</div>"
       "<div class='step-text'>Ils étudient les marchés, comparent des répartitions et contrôlent les risques.</div></div>"
       "<div class='step' style='--c:#D9A441'><div class='step-num'>ÉTAPE 3</div>"
       "<div class='step-title'>Vous recevez une proposition</div>"
       "<div class='step-text'>Une répartition chiffrée de votre capital, expliquée en langage simple.</div></div>"
       "</div>")

    dates = [d for d in reversed(list(rets.index[config.ESTIMATION_WINDOW - 1:]))]
    with st.form("profil"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("##### Votre projet")
            first_name = st.text_input("Prénom", placeholder="Facultatif")
            capital_txt = st.text_input("Capital à investir, en dirhams", placeholder="Par exemple 100 000")
            horizon = st.selectbox("Durée prévue de l'investissement", HORIZONS, index=None,
                                   placeholder="Choisissez une durée")
            objective = st.selectbox("Votre objectif", list(OBJECTIVES), index=None,
                                     placeholder="Choisissez un objectif")
        with right:
            st.markdown("##### Votre rapport au risque")
            risk = st.radio("Votre profil de risque", list(RISKS), index=None,
                            captions=list(RISKS.values()))
            reaction = st.selectbox("Si votre portefeuille perdait 15 % en quelques mois, que feriez-vous ?",
                                    REACTIONS, index=None, placeholder="Choisissez une réponse")
            market = st.selectbox("Marché souhaité", MARKETS, index=None, placeholder="Choisissez un marché")
        with st.expander("Réglages avancés"):
            a1, a2 = st.columns(2)
            as_of_choice = a1.selectbox("Date d'analyse", dates, index=None, format_func=fr_month,
                                        placeholder="Dernière date disponible")
            mode_label = a2.selectbox("Méthode de choix final", list(MODES), index=None,
                                      placeholder="Recommandée")
        submitted = st.form_submit_button("Voir ma proposition", type="primary")

    if not submitted:
        return
    missing = []
    capital = None
    if not capital_txt.strip():
        missing.append("le capital à investir")
    else:
        capital = parse_capital(capital_txt)
        if capital is None or not (1000 <= capital <= 50_000_000):
            missing.append("un capital valide (entre 1 000 et 50 000 000 DH)")
    for value, label in [(horizon, "la durée"), (objective, "l'objectif"), (risk, "le profil de risque"),
                         (reaction, "votre réaction face à une baisse"), (market, "le marché")]:
        if value is None:
            missing.append(label)
    if missing:
        st.error("Merci de compléter : " + ", ".join(missing) + ".")
        return

    agents.SELECTION_MODE = MODES.get(mode_label, "consensus")
    profile = agents.InvestorProfile(capital=float(capital), risk=RISK_KEYS[risk],
                                     objective=OBJECTIVES[objective], horizon=horizon)
    with st.spinner("Nos agents analysent les marchés..."):
        ctx = agents.run_assistant(profile, as_of=pd.Timestamp(as_of_choice) if as_of_choice is not None else None)
    st.session_state["ctx"] = ctx
    st.session_state["answers"] = {"first_name": first_name.strip(), "reaction": reaction, "market": market}
    st.rerun()


def results_page(ctx, rets: pd.DataFrame, answers: dict) -> None:
    s, p = ctx.selected, ctx.profile
    name = html.escape(answers["first_name"])
    head, back = st.columns([4, 1])
    with head:
        title = f"Bonjour {name}, voici notre proposition" if name else "Voici notre proposition"
        md(f"<p class='page-title'>{title}</p>"
           f"<p class='page-sub'>Capital de {dh(p.capital)}, profil {RISK_LABELS[p.risk].lower()}. "
           f"Analyse des marchés arrêtée à {fr_month(ctx.as_of)}.</p>")
    with back:
        if st.button("Modifier mes réponses"):
            st.session_state.pop("ctx", None)
            st.rerun()

    for kind, text in notices(ctx, answers):
        (st.warning if kind == "warning" else st.info)(text)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.markdown(kpi("Gain annuel estimé", fr_pct(s.exp_return),
                    "Estimation à partir des cinq dernières années, sans garantie.", KPI_COLORS["gain"]),
                unsafe_allow_html=True)
    k2.markdown(kpi("Variation annuelle courante", fr_pct(s.vol),
                    f"En année normale, la valeur peut monter ou baisser d'environ {dh(s.vol * p.capital)}.",
                    KPI_COLORS["variation"]), unsafe_allow_html=True)
    if s.worst_month < 0:
        k3.markdown(kpi("Plus forte baisse sur un mois", fr_pct(s.worst_month),
                        f"Observée ces cinq dernières années, soit environ {dh(abs(s.worst_month) * p.capital)}.",
                        KPI_COLORS["baisse"]), unsafe_allow_html=True)
    else:
        k3.markdown(kpi("Plus forte baisse sur un mois", "Aucune",
                        "Pas de mois en baisse sur les cinq dernières années.", KPI_COLORS["baisse"]),
                    unsafe_allow_html=True)
    if s.cash > 0.005:
        k4.markdown(kpi("Part gardée en liquidités", fr_pct(s.cash, 0),
                        f"{dh(s.cash * p.capital)} restent disponibles et rapportent le taux sans risque.",
                        KPI_COLORS["liquidites"]), unsafe_allow_html=True)
    else:
        k4.markdown(kpi("Part gardée en liquidités", "0\u00a0%", "Tout votre capital est investi.",
                        KPI_COLORS["liquidites"]), unsafe_allow_html=True)

    md("<div class='section-title'>Répartition de votre capital</div>")
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        st.plotly_chart(donut(ctx), config={"displayModeBar": False})
    with c2:
        md(alloc_html(ctx))
    with st.expander("Voir le détail fonds par fonds"):
        rows = [{"Fonds": FUND_LABELS[t], "Code": t, "Part": fr_pct(w), "Montant": dh(w * p.capital)}
                for t, w in sorted(zip(ctx.assets, s.weights), key=lambda x: -x[1]) if w > 0.005]
        if s.cash > 0.005:
            rows.append({"Fonds": "Liquidités", "Code": "", "Part": fr_pct(s.cash), "Montant": dh(s.cash * p.capital)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    md("<div class='section-title'>Ce qu'il faut retenir</div>")
    md("<div class='card'><ul class='points'>" + "".join(f"<li>{t}</li>" for t in takeaways(ctx)) + "</ul></div>")

    md("<div class='section-title'>Pour aller plus loin</div>")
    tabs = st.tabs(["Performance passée", "Crises passées", "Répartitions comparées", "Notre démarche"])
    with tabs[0]:
        st.plotly_chart(growth_chart(ctx), config={"displayModeBar": False})
        st.caption("Ce que ce portefeuille aurait donné sur les cinq dernières années. C'est une simulation "
                   "rétrospective, pas un résultat garanti.")
    with tabs[1]:
        st.write("Ce que cette répartition aurait donné pendant trois épisodes difficiles des marchés.")
        st.dataframe(stress_table(ctx, rets).set_index("Épisode").style.map(color_signed))
        mix = profiles.REFERENCE_MIXES[p.risk]
        st.caption(f"Le mélange classique représente {fr_pct(mix.get('SPY', 0), 0)} d'actions américaines et "
                   f"{fr_pct(mix.get('AGG', 0), 0)} d'obligations.")
    with tabs[2]:
        st.write("Les différentes façons de répartir votre capital que nos agents ont étudiées, "
                 "et ce que le contrôle des risques en a fait.")
        st.dataframe(approaches_table(ctx).style.map(color_result, subset=["Résultat"]), hide_index=True)
    with tabs[3]:
        st.write("Sept agents se sont relayés pour construire cette proposition.")
        md(agents_html(agent_steps(ctx)))
        with st.expander("Journal détaillé des agents"):
            for sender, receiver, text in ctx.trace:
                st.markdown(f"**{sender}** vers *{receiver}* : {text}")

    md("<div class='legal'>Les performances passées ne préjugent pas des performances futures. Les montants sont "
       "indicatifs. Les fonds proposés sont des ETF cotés en dollars aux États-Unis : l'effet de change avec le "
       "dirham n'est pas pris en compte.</div>")


def main() -> None:
    st.set_page_config(page_title="Assistant de portefeuille", layout="wide")
    md(CSS)
    rets = load_returns()
    ctx = st.session_state.get("ctx")
    if ctx is None:
        form_page(rets)
    else:
        results_page(ctx, rets, st.session_state["answers"])


main()
