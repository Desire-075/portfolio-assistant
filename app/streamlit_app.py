"""Application Streamlit : assistant multi-agent de selection de portefeuilles."""
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

PROFILE_NAMES = {"Prudent": "Prudent", "Equilibre": "Équilibré", "Dynamique": "Dynamique"}
RISK_LABELS = {
    "Prudent": "Prudent : je veux limiter les pertes potentielles",
    "Equilibre": "Équilibré : j'accepte un risque raisonnable",
    "Dynamique": "Dynamique : je recherche un rendement plus élevé et j'accepte davantage de risques",
}
OBJ_LABELS = {
    "Preserver": "Préserver mon capital",
    "Fructifier": "Faire fructifier mon capital",
    "Maximiser": "Maximiser le rendement à long terme",
}
HORIZONS = ["Moins d'un an", "1 à 3 ans", "3 à 5 ans", "Plus de 5 ans"]
STRESS = {
    "Crise financière (sept. 2008 - févr. 2009)": ("2008-09-01", "2009-02-28"),
    "Covid (févr. - mars 2020)": ("2020-02-01", "2020-03-31"),
    "Choc de taux (année 2022)": ("2022-01-01", "2022-12-31"),
}


def dh(x: float) -> str:
    return f"{x:,.0f} DH".replace(",", " ")


@st.cache_data
def load_returns() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "returns_monthly.csv"
    if not path.exists():           # permet un deploiement a partir des seules donnees brutes
        prepare_data.main()
    return pd.read_csv(path, index_col=0, parse_dates=True)


def reference_weights(ctx) -> np.ndarray:
    mix = profiles.REFERENCE_MIXES[ctx.profile.risk]
    return np.array([mix.get(t, 0.0) for t in ctx.assets])


def by_class(ctx) -> dict:
    out = {}
    for t, w in zip(ctx.assets, ctx.selected.weights):
        cls = config.UNIVERSE[t][1]
        out[cls] = out.get(cls, 0.0) + float(w)
    out["Liquidités"] = ctx.selected.cash
    return {k: v for k, v in out.items() if v > 0.005}


def profile_warnings(p) -> list:
    msgs = []
    if p.horizon == "Moins d'un an" and p.risk == "Dynamique":
        msgs.append("Un horizon inférieur à un an est peu compatible avec un profil Dynamique : "
                    "les pertes temporaires pourraient ne pas avoir le temps d'être récupérées.")
    if p.objective == "Maximiser" and p.risk == "Prudent":
        msgs.append("L'objectif de maximisation du rendement est en tension avec un profil Prudent : "
                    "la contrainte de risque du profil l'emporte.")
    return msgs


def growth_chart(ctx) -> go.Figure:
    hist, s = ctx.hist, ctx.selected
    series = {
        "Portefeuille proposé": hist[ctx.assets].values @ s.weights + s.cash * hist["RF"].values,
        "Mélange de référence": hist[ctx.assets].values @ reference_weights(ctx),
        "SPY (actions US)": hist["SPY"].values,
    }
    fig = go.Figure()
    for name, r in series.items():
        fig.add_trace(go.Scatter(x=hist.index, y=100 * np.cumprod(1 + r), name=name, mode="lines"))
    fig.update_layout(yaxis_title="Valeur (base 100)", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    return fig


def stress_table(ctx, rets: pd.DataFrame) -> pd.DataFrame:
    s, ref_w = ctx.selected, reference_weights(ctx)
    rows = []
    for name, (a, b) in STRESS.items():
        r = rets.loc[a:b]
        if r.empty:
            continue
        port = r[ctx.assets].values @ s.weights + s.cash * r["RF"].values
        ref = r[ctx.assets].values @ ref_w
        rows.append({"Scénario": name,
                     "Portefeuille proposé": (1 + port).prod() - 1,
                     "Mélange de référence": (1 + ref).prod() - 1,
                     "SPY": (1 + r["SPY"]).prod() - 1})
    df = pd.DataFrame(rows).set_index("Scénario")
    return df.map(lambda x: f"{x:+.1%}")


def candidates_table(ctx) -> pd.DataFrame:
    cands = list(ctx.candidates)
    if ctx.selected not in cands:
        cands.append(ctx.selected)
    rows = []
    for c in cands:
        retained = c is ctx.selected
        rows.append({
            "Méthode": c.name + (" (retenu)" if retained else ""),
            "Volatilité": f"{c.vol:.1%}",
            "Rendement attendu": f"{c.exp_return:.1%}",
            "Sharpe attendu": f"{c.sharpe:.2f}",
            "Pire mois": f"{c.worst_month:.1%}",
            "Positions effectives": f"{c.eff_n:.1f}",
            "Décision": "Accepté" if c.accepted else "Écarté",
            "Raisons": "; ".join(c.reasons),
        })
    return pd.DataFrame(rows)


def render(ctx, rets: pd.DataFrame) -> None:
    s, p = ctx.selected, ctx.profile
    st.subheader(f"Portefeuille proposé : profil {PROFILE_NAMES[p.risk]} "
                 f"(décision au {ctx.as_of:%d/%m/%Y})")
    for msg in profile_warnings(p):
        st.warning(msg)
    if not any(c.accepted for c in ctx.candidates):
        st.warning(ctx.selection_note)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rendement annuel attendu (indicatif)", f"{s.exp_return:.1%}")
    m2.metric("Volatilité annuelle estimée", f"{s.vol:.1%}", f"cible {ctx.target_vol:.1%}", delta_color="off")
    m3.metric("Pire mois (60 derniers mois)", f"{s.worst_month:.1%}")
    m4.metric("Positions effectives", f"{s.eff_n:.1f}")

    left, right = st.columns(2)
    classes = by_class(ctx)
    donut = go.Figure(go.Pie(labels=list(classes), values=list(classes.values()), hole=0.5,
                             textinfo="label+percent"))
    donut.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False, height=360)
    left.plotly_chart(donut)
    rows = [{"ETF": t, "Nom": config.UNIVERSE[t][0], "Classe": config.UNIVERSE[t][1],
             "Poids": f"{w:.1%}", "Montant": dh(w * p.capital)}
            for t, w in sorted(zip(ctx.assets, s.weights), key=lambda x: -x[1]) if w > 0.005]
    if s.cash > 0.005:
        rows.append({"ETF": "-", "Nom": "Liquidités (taux sans risque)", "Classe": "Liquidités",
                     "Poids": f"{s.cash:.1%}", "Montant": dh(s.cash * p.capital)})
    right.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("### Pourquoi cette proposition ?")
    n_acc = sum(c.accepted for c in ctx.candidates)
    lines = [f"**Méthode retenue : {s.name}.** {ctx.selection_note}",
             f"La volatilité estimée ({s.vol:.1%}) vise la cible du profil ({ctx.target_vol:.1%})"
             + (f" : {s.cash:.0%} du capital est placé en liquidités pour y parvenir." if s.cash > 0.005 else "."),
             f"{n_acc} candidat(s) sur {len(ctx.candidates)} respectent les règles de risque "
             "(bande de volatilité, poids maximal par ETF, diversification minimale)."]
    for line in lines:
        st.markdown(f"- {line}")

    tabs = st.tabs(["Historique simulé", "Scénarios de stress", "Candidats évalués",
                    "Trace des agents", "Explication complète"])
    with tabs[0]:
        st.plotly_chart(growth_chart(ctx))
        st.caption("Simulation rétrospective sur les 60 derniers mois avec les poids proposés : "
                   "ce n'est pas une performance réelle et elle ne garantit rien pour l'avenir.")
    with tabs[1]:
        st.dataframe(stress_table(ctx, rets))
        st.caption("Rejeu de périodes historiques avec les poids proposés (indicatif).")
    with tabs[2]:
        st.dataframe(candidates_table(ctx), hide_index=True)
    with tabs[3]:
        for sender, receiver, text in ctx.trace:
            st.markdown(f"**{sender}** → *{receiver}* : {text}")
    with tabs[4]:
        st.text(ctx.explanation)

    st.caption("Exercice pédagogique, pas un conseil en investissement. Univers : 11 ETF américains cotés en "
               "dollars ; la conversion en dirhams et le marché marocain ne sont pas modélisés.")


def main() -> None:
    st.set_page_config(page_title="Assistant de portefeuille", layout="wide")
    rets = load_returns()
    st.title("Assistant multi-agent de sélection de portefeuille")
    st.caption("Renseignez votre profil : des agents spécialisés analysent les marchés, évaluent des "
               "portefeuilles candidats, contrôlent le risque et expliquent leur choix.")

    with st.sidebar:
        st.header("Mon profil d'investisseur")
        capital = st.number_input("Capital à investir (DH)", min_value=1000, value=100000, step=1000)
        horizon = st.selectbox("Durée prévue de l'investissement", HORIZONS, index=2)
        risk = st.radio("Profil de risque", list(RISK_LABELS), format_func=RISK_LABELS.get)
        objective = st.radio("Objectif", list(OBJ_LABELS), index=1, format_func=OBJ_LABELS.get)
        st.selectbox("Marché", ["International (ETF américains cotés)"], disabled=True,
                     help="Le marché marocain n'est pas couvert par les données de ce projet.")
        with st.expander("Options de démonstration"):
            dates = [d.strftime("%Y-%m-%d") for d in rets.index[config.ESTIMATION_WINDOW - 1:]]
            as_of = st.select_slider("Date de décision", options=dates, value=dates[-1])
            mode = st.selectbox("Sélection finale", ["consensus", "best"],
                                format_func=lambda m: "Consensus (recommandé)" if m == "consensus"
                                else "Meilleur candidat (v1)")
        clicked = st.button("Générer mon portefeuille", type="primary")

    if clicked:
        agents.SELECTION_MODE = mode
        profile = agents.InvestorProfile(capital=float(capital), risk=risk, objective=objective, horizon=horizon)
        with st.spinner("Les agents analysent les marchés..."):
            st.session_state["ctx"] = agents.run_assistant(profile, as_of=pd.Timestamp(as_of))

    ctx = st.session_state.get("ctx")
    if ctx is None:
        st.info("Renseignez votre profil dans le panneau de gauche, puis cliquez sur « Générer mon portefeuille ».")
        return
    render(ctx, rets)


main()
