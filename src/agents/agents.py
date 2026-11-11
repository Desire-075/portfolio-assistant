"""Assistant multi-agent de selection de portefeuilles.

Chaque agent lit le contexte partage, ecrit sa sortie et laisse une trace (qui transmet quoi a qui).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src import config, profiles
from src.optimization import allocators as al
from src.prepare_data import split

VOL_BAND = (0.75, 1.05)      # bande de volatilite acceptee, en multiple de la cible effective
MAX_ASSET_WEIGHT = 0.40      # poids maximal d'un ETF
MIN_EFFECTIVE_N = 3.0        # nombre minimal de positions effectives (1 / somme des carres)
SHRINKAGE = 0.5              # rendement attendu = 50 % historique + 50 % moyenne de l'univers


@dataclass
class InvestorProfile:
    capital: float
    risk: str          # "Prudent" | "Equilibre" | "Dynamique"
    objective: str     # "Preserver" | "Fructifier" | "Maximiser"
    horizon: str = "3 a 5 ans"


@dataclass
class Candidate:
    name: str
    weights: np.ndarray            # poids sur les ETF ; le reste est en liquidites
    cash: float = 0.0
    vol: float = 0.0               # volatilite annualisee ex ante
    exp_return: float = 0.0        # rendement annualise attendu (avec retrecissement)
    sharpe: float = 0.0
    worst_month: float = 0.0       # pire mois sur la fenetre historique
    eff_n: float = 0.0
    max_weight: float = 0.0
    accepted: bool = False
    reasons: list = field(default_factory=list)


@dataclass
class Context:
    profile: InvestorProfile
    as_of: pd.Timestamp | None = None
    rets_all: pd.DataFrame | None = None
    assets: list = field(default_factory=list)
    hist: pd.DataFrame | None = None
    mu: np.ndarray | None = None
    cov: np.ndarray | None = None
    rf: float = 0.0
    target_vol: float = 0.0        # cible effective (marge de securite incluse)
    analysis: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    selected: Candidate | None = None
    selection_note: str = ""
    explanation: str = ""
    trace: list = field(default_factory=list)

    def log(self, sender: str, receiver: str, text: str) -> None:
        self.trace.append((sender, receiver, text))


class DataAgent:
    name = "Agent Donnees"

    def run(self, ctx: Context) -> None:
        rets = pd.read_csv(config.DATA_PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
        as_of = ctx.as_of if ctx.as_of is not None else rets.index.max()
        hist = rets.loc[:as_of].tail(config.ESTIMATION_WINDOW)   # rien de posterieur a la date de decision
        if len(hist) < config.ESTIMATION_WINDOW:
            raise ValueError("Historique insuffisant pour cette date")
        ctx.rets_all = rets
        ctx.as_of = hist.index.max()
        ctx.assets = [c for c in rets.columns if c != "RF"]
        ctx.hist = hist
        ctx.mu, ctx.cov = al.estimate(hist[ctx.assets])
        ctx.rf = float(hist["RF"].mean())
        missing = int(hist.isna().sum().sum())
        ctx.log(self.name, "Agent Profil",
                f"Fenetre {hist.index.min().date()} -> {hist.index.max().date()} ({len(hist)} mois, "
                f"{missing} valeur manquante), mu et covariance (Ledoit-Wolf), "
                f"taux sans risque {ctx.rf * 12:.2%}")


class ProfileAgent:
    name = "Agent Profil"

    def run(self, ctx: Context) -> None:
        train, _, _ = split(ctx.rets_all)
        base = profiles.target_vols(train[ctx.assets])[ctx.profile.risk]   # cible fixee a priori sur le train
        ctx.target_vol = config.VOL_BUFFER * base
        p = ctx.profile
        ctx.log(self.name, "Agent Analyse",
                f"Profil {p.risk}, objectif {p.objective}, horizon {p.horizon} : volatilite de reference "
                f"{base:.1%}, cible effective {ctx.target_vol:.1%} (marge {config.VOL_BUFFER})")


class AnalysisAgent:
    name = "Agent Analyse"

    def run(self, ctx: Context) -> None:
        a = ctx.hist[ctx.assets]
        stats = pd.DataFrame({
            "Rendement ann.": a.mean() * 12,
            "Volatilite ann.": a.std() * np.sqrt(12),
        })
        stats["Sharpe"] = a.sub(ctx.hist["RF"], axis=0).mean() / a.std() * np.sqrt(12)
        corr = a.corr().values
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        link = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(link, t=3, criterion="maxclust")
        clusters = {int(c): [ctx.assets[i] for i in np.where(labels == c)[0]] for c in sorted(set(labels))}
        avg_corr = float(corr[np.triu_indices_from(corr, k=1)].mean())
        ctx.analysis = {"stats": stats, "clusters": clusters, "avg_corr": avg_corr}
        groups = " | ".join(", ".join(v) for v in clusters.values())
        ctx.log(self.name, "Agent Optimisation",
                f"Correlation moyenne {avg_corr:.2f} ; 3 groupes d'actifs : {groups}")


class OptimizationAgent:
    name = "Agent Optimisation"

    def run(self, ctx: Context) -> None:
        n, mw, cov, mu, rf = len(ctx.assets), config.MAX_WEIGHT, ctx.cov, ctx.mu, ctx.rf
        base = {
            "1/N": lambda: al.equal_weight(n),
            "Vol inverse": lambda: al.inverse_vol(cov),
            "Min variance": lambda: al.min_variance(cov, mw),
            "Max Sharpe": lambda: al.max_sharpe(mu, cov, rf, mw),
            "HRP": lambda: al.hrp(cov),
        }
        ctx.candidates = []
        for name, fn in base.items():
            try:
                w = fn()
            except RuntimeError as e:
                ctx.log(self.name, "Agent Risque", f"{name} : optimisation echouee ({e})")
                continue
            w = profiles.scale_to_target(w, cov, ctx.target_vol, buffer=1.0)
            ctx.candidates.append(Candidate(name=name, weights=w, cash=max(0.0, 1.0 - float(w.sum()))))
        ctx.log(self.name, "Agent Risque",
                f"{len(ctx.candidates)} candidats ramenes vers les liquidites pour viser {ctx.target_vol:.1%}")


class RiskAgent:
    name = "Agent Risque"

    def assess(self, ctx: Context, c: Candidate) -> Candidate:
        """Calcule les indicateurs d'un candidat et applique les regles de risque."""
        mu_s = SHRINKAGE * ctx.mu + (1 - SHRINKAGE) * ctx.mu.mean()
        hist_assets = ctx.hist[ctx.assets].values
        rf_hist = ctx.hist["RF"].values
        lo, hi = VOL_BAND
        w = c.weights
        c.vol = float(np.sqrt(w @ ctx.cov @ w * 12))
        c.exp_return = float((w @ mu_s + c.cash * ctx.rf) * 12)
        c.sharpe = (c.exp_return - ctx.rf * 12) / c.vol if c.vol > 0 else 0.0
        c.worst_month = float((hist_assets @ w + c.cash * rf_hist).min())
        c.max_weight = float(w.max())
        c.eff_n = float(1.0 / np.sum((w / w.sum()) ** 2))
        c.reasons = []
        if c.vol < lo * ctx.target_vol:
            c.reasons.append(f"volatilite {c.vol:.1%} sous la bande (min {lo * ctx.target_vol:.1%})")
        if c.vol > hi * ctx.target_vol:
            c.reasons.append(f"volatilite {c.vol:.1%} au-dessus de la bande (max {hi * ctx.target_vol:.1%})")
        if c.max_weight > MAX_ASSET_WEIGHT + 1e-9:
            c.reasons.append(f"un ETF depasse {MAX_ASSET_WEIGHT:.0%} ({c.max_weight:.0%})")
        if c.eff_n < MIN_EFFECTIVE_N:
            c.reasons.append(f"diversification insuffisante ({c.eff_n:.1f} positions effectives)")
        c.accepted = not c.reasons
        return c

    def run(self, ctx: Context) -> None:
        for c in ctx.candidates:
            self.assess(ctx, c)
        ok = [c.name for c in ctx.candidates if c.accepted]
        ctx.log(self.name, "Agent Selection",
                f"{len(ok)}/{len(ctx.candidates)} candidats acceptes : {', '.join(ok) if ok else 'aucun'}")


# "best" : un seul candidat (v1) ; "consensus" : moyenne des candidats acceptes (v2)
SELECTION_MODE = "consensus"


class SelectionAgent:
    name = "Agent Selection"

    def run(self, ctx: Context) -> None:
        acc = [c for c in ctx.candidates if c.accepted]
        obj = ctx.profile.objective
        if not acc:
            best = min(ctx.candidates, key=lambda c: abs(c.vol - ctx.target_vol))
            ctx.selection_note = ("Aucun candidat ne respecte toutes les regles de risque ; "
                                  "retenu : le plus proche de la volatilite cible (a examiner avec prudence).")
        elif SELECTION_MODE == "consensus" and len(acc) > 1:
            w = np.mean([c.weights for c in acc], axis=0)
            best = Candidate(name="Consensus", weights=w, cash=max(0.0, 1.0 - float(w.sum())))
            RiskAgent().assess(ctx, best)
            ctx.selection_note = (f"Consensus : moyenne des {len(acc)} candidats acceptes "
                                  f"({', '.join(c.name for c in acc)}), ce qui limite le risque de modele et la rotation.")
        elif obj == "Preserver":
            best = max(acc, key=lambda c: c.worst_month)
            ctx.selection_note = "Objectif de preservation : pire mois historique le moins mauvais parmi les candidats acceptes."
        elif obj == "Fructifier":
            best = max(acc, key=lambda c: c.sharpe)
            ctx.selection_note = "Objectif de croissance : meilleur rapport rendement/risque attendu parmi les candidats acceptes."
        else:
            best = max(acc, key=lambda c: c.exp_return)
            ctx.selection_note = "Objectif de rendement : rendement attendu le plus eleve parmi les candidats acceptes."
        ctx.selected = best
        ctx.log(self.name, "Agent Synthese", f"Retenu : {best.name}. {ctx.selection_note}")


class ExplainAgent:
    name = "Agent Synthese"

    def run(self, ctx: Context) -> None:
        s, p = ctx.selected, ctx.profile
        by_class = {}
        for t, w in zip(ctx.assets, s.weights):
            cls = config.UNIVERSE[t][1]
            by_class[cls] = by_class.get(cls, 0.0) + float(w)
        by_class["Liquidites"] = s.cash

        def money(x: float) -> str:
            return f"{x * p.capital:,.0f} DH".replace(",", " ")

        lines = [
            f"PORTEFEUILLE PROPOSE - profil {p.risk}, objectif {p.objective}, "
            f"capital {p.capital:_.0f} DH (decision au {ctx.as_of.date()})".replace("_", " "),
            f"Methode retenue : {s.name}. {ctx.selection_note}",
            "",
            "Repartition par classe d'actifs :",
        ]
        for cls, w in sorted(by_class.items(), key=lambda kv: -kv[1]):
            if w > 0.005:
                lines.append(f"  {cls:<26}{w:6.1%}   {money(w)}")
        lines += ["", "Detail par ETF :"]
        order = np.argsort(-s.weights)
        for i in order:
            if s.weights[i] > 0.005:
                t = ctx.assets[i]
                lines.append(f"  {t:<5}{config.UNIVERSE[t][0]:<34}{s.weights[i]:6.1%}   {money(s.weights[i])}")
        lines += [
            "",
            f"Estimations ex ante (historique des {config.ESTIMATION_WINDOW} derniers mois) :",
            f"  Rendement annuel attendu (indicatif) : {s.exp_return:.1%}",
            f"  Volatilite annuelle estimee          : {s.vol:.1%}  (cible {ctx.target_vol:.1%})",
            f"  Pire mois sur la fenetre             : {s.worst_month:.1%}",
            f"  Positions effectives                 : {s.eff_n:.1f}",
            "",
            "Candidats ecartes par l'agent de risque :",
        ]
        rejected = [c for c in ctx.candidates if not c.accepted]
        if rejected:
            for c in rejected:
                lines.append(f"  {c.name} : {'; '.join(c.reasons)}")
        else:
            lines.append("  aucun")
        lines += [
            "",
            "Limites : estimations historiques, sans garantie de rendement futur ; univers d'ETF americains "
            "cotes en dollars (conversion en DH non modelisee) ; exercice pedagogique, pas un conseil en investissement.",
        ]
        ctx.explanation = "\n".join(lines)
        ctx.log(self.name, "Client", "Proposition et explication generees")


def run_assistant(profile: InvestorProfile, as_of: pd.Timestamp | None = None) -> Context:
    ctx = Context(profile=profile, as_of=as_of)
    for agent in [DataAgent(), ProfileAgent(), AnalysisAgent(), OptimizationAgent(),
                  RiskAgent(), SelectionAgent(), ExplainAgent()]:
        agent.run(ctx)
    return ctx


if __name__ == "__main__":
    first = None
    for risk in ["Prudent", "Equilibre", "Dynamique"]:
        ctx = run_assistant(InvestorProfile(capital=100_000, risk=risk, objective="Fructifier"))
        first = first or ctx
        print("=" * 100)
        print(ctx.explanation)
    print("=" * 100)
    print("TRACE DES AGENTS (profil Prudent) :")
    for sender, receiver, text in first.trace:
        print(f"  [{sender}] -> [{receiver}] : {text}")


