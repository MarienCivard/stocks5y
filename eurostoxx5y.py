#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:15:05 2025

@author: mariencivard
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eurostoxx5y.py — Calcule la moyenne du rendement sur 5 ans (glissant) de l'EURO STOXX 50
depuis la première date disponible sur Yahoo Finance (indice prix par défaut).

Usage:
    python eurostoxx5y.py --ticker "^STOXX50E" --plot
    # ou total return (si dispo sur Yahoo)
    python eurostoxx5y.py --ticker "^SX5T"

Dépendances:
    pip install yfinance pandas matplotlib
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eurostoxx5y.py — Calcule la moyenne du rendement sur 5 ans (glissant) de l'EURO STOXX 50
depuis la première date disponible sur Yahoo Finance (indice prix par défaut).

Usage:
    python eurostoxx5y.py --ticker "^STOXX50E" --plot
    # ou total return (si dispo sur Yahoo)
    python eurostoxx5y.py --ticker "^SX5T"

Dépendances:
    pip install yfinance pandas matplotlib
"""

import math
import argparse
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FiveYearStats:
    start: pd.Timestamp
    end: pd.Timestamp
    n_windows_rolling: int
    n_windows_block: int
    avg_5y_cagr_rolling: float       # moyenne arithmétique des CAGR 5 ans (fenêtres glissantes)
    med_5y_cagr_rolling: float
    avg_5y_totalret_rolling: float   # moyenne arithmétique des rendements cumulés 5 ans (glissant)
    med_5y_totalret_rolling: float
    avg_5y_cagr_block: float         # mêmes métriques en fenêtres non chevauchantes
    med_5y_cagr_block: float
    avg_5y_totalret_block: float
    med_5y_totalret_block: float


def _coerce_to_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Garantit une Series 1D float (squeeze si DataFrame)."""
    if isinstance(x, pd.DataFrame):
        x = x.squeeze("columns")
    if not isinstance(x, pd.Series):
        raise TypeError("px_monthly doit être une pandas Series.")
    return pd.to_numeric(x, errors="coerce")


def download_index_prices(ticker: str = "^STOXX50E") -> pd.Series:
    """
    Télécharge l'historique quotidien depuis Yahoo Finance et renvoie une série
    mensuelle (fin de mois) des prix ajustés (ou clôture si Adj Close absent).
    """
    df = yf.download(
        tickers=ticker,
        period="max",
        interval="1d",
        auto_adjust=True,     # pour un indice, équivaut à Close
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"Aucune donnée récupérée pour {ticker}.")

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        # Certains indices ne renvoient qu'une seule colonne nommée comme le ticker
        if ticker in df.columns:
            price_col = ticker
        else:
            raise RuntimeError(f"Colonne prix introuvable dans les données pour {ticker}.")

    px_daily = df[price_col].dropna()

    # Fin de mois: utiliser "ME" (month-end) — "M" est déprécié
    px_monthly = px_daily.resample("ME").last().dropna()

    # S'assurer d'une Series float
    px_monthly = _coerce_to_series(px_monthly)
    px_monthly.name = ticker
    return px_monthly


def compute_5y_metrics(px_monthly: pd.Series) -> FiveYearStats:
    """
    Calcule les rendements 5 ans glissants et par blocs non chevauchants,
    ainsi que leurs statistiques (moyenne/médiane).
    """
    px_monthly = _coerce_to_series(px_monthly).dropna()
    if len(px_monthly) < 61:  # besoin d'au moins 60 mois + 1 point
        raise RuntimeError(
            "Pas assez d'historique pour former au moins une fenêtre de 5 ans (>= 61 points mensuels)."
        )

    # --- Fenêtres glissantes (60 mois) ---
    ratio_60m = px_monthly / px_monthly.shift(60)
    totalret_5y_roll = (ratio_60m - 1.0).dropna()
    cagr_5y_roll = (ratio_60m ** (1 / 5) - 1.0).dropna()

    # --- Fenêtres non chevauchantes (tous les 60 mois), robustes ---
    vals = px_monthly.to_numpy(dtype=float)
    block_ratios = []
    for i in range(60, len(vals), 60):  # i = 60, 120, 180, ...
        start_px = vals[i - 60]
        end_px = vals[i]
        if np.isfinite(start_px) and np.isfinite(end_px) and start_px > 0.0:
            block_ratios.append(end_px / start_px)

    block_ratios = pd.Series(block_ratios, dtype=float)
    totalret_5y_block = block_ratios - 1.0
    cagr_5y_block = block_ratios ** (1 / 5) - 1.0

    return FiveYearStats(
        start=px_monthly.index[0],
        end=px_monthly.index[-1],
        n_windows_rolling=int(cagr_5y_roll.shape[0]),
        n_windows_block=int(cagr_5y_block.shape[0]),
        avg_5y_cagr_rolling=float(cagr_5y_roll.mean()),
        med_5y_cagr_rolling=float(cagr_5y_roll.median()),
        avg_5y_totalret_rolling=float(totalret_5y_roll.mean()),
        med_5y_totalret_rolling=float(totalret_5y_roll.median()),
        avg_5y_cagr_block=float(cagr_5y_block.mean()) if len(cagr_5y_block) else float("nan"),
        med_5y_cagr_block=float(cagr_5y_block.median()) if len(cagr_5y_block) else float("nan"),
        avg_5y_totalret_block=float(totalret_5y_block.mean()) if len(totalret_5y_block) else float("nan"),
        med_5y_totalret_block=float(totalret_5y_block.median()) if len(totalret_5y_block) else float("nan"),
    )


def format_pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "n/a"


def main(ticker: str = "^STOXX50E", plot: bool = False) -> FiveYearStats:
    px_m = download_index_prices(ticker=ticker)
    stats = compute_5y_metrics(px_m)

    print(f"Indice: {ticker}")
    print(f"Période disponible: {stats.start.date()} → {stats.end.date()}")
    print(f"Nombre de fenêtres glissantes 5 ans: {stats.n_windows_rolling}")
    print("— Rendements 5 ans (fenêtres glissantes) —")
    print(f"  • CAGR moyen (annualisé): {format_pct(stats.avg_5y_cagr_rolling)}")
    print(f"  • CAGR médian (annualisé): {format_pct(stats.med_5y_cagr_rolling)}")
    print(f"  • Rendement cumulé moyen sur 5 ans: {format_pct(stats.avg_5y_totalret_rolling)}")
    print(f"  • Rendement cumulé médian sur 5 ans: {format_pct(stats.med_5y_totalret_rolling)}")

    print(f"\nFenêtres non chevauchantes (tous les 60 mois) : {stats.n_windows_block} bloc(s)")
    print(
        f"  • CAGR moyen (annualisé): "
        f"{format_pct(stats.avg_5y_cagr_block) if not math.isnan(stats.avg_5y_cagr_block) else 'n/a'}"
    )
    print(
        f"  • CAGR médian (annualisé): "
        f"{format_pct(stats.med_5y_cagr_block) if not math.isnan(stats.med_5y_cagr_block) else 'n/a'}"
    )
    print(
        f"  • Rendement cumulé moyen sur 5 ans: "
        f"{format_pct(stats.avg_5y_totalret_block) if not math.isnan(stats.avg_5y_totalret_block) else 'n/a'}"
    )
    print(
        f"  • Rendement cumulé médian sur 5 ans: "
        f"{format_pct(stats.med_5y_totalret_block) if not math.isnan(stats.med_5y_totalret_block) else 'n/a'}"
    )

    if plot:
        try:
            import matplotlib.pyplot as plt

            ratio_60m = px_m / px_m.shift(60)
            cagr_5y_roll = (ratio_60m ** (1 / 5) - 1.0).dropna()

            plt.figure(figsize=(10, 4))
            cagr_5y_roll.plot(title=f"CAGR annualisé sur 5 ans ({ticker}) – fenêtres glissantes")
            plt.ylabel("CAGR 5 ans")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Avertissement] Impossible d'afficher le graphique: {e}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAGR moyen sur 5 ans de l'EURO STOXX 50 (Yahoo Finance).")
    parser.add_argument("--ticker", type=str, default="^STOXX50E", help="Ticker Yahoo (ex: ^STOXX50E, ^SX5T)")
    parser.add_argument("--plot", action="store_true", help="Afficher le graphique du CAGR 5 ans glissant")
    args = parser.parse_args()

    main(ticker=args.ticker, plot=args.plot)

