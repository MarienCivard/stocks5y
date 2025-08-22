# eurostoxx5y

Script Python qui calcule le **rendement sur 5 ans** de l’EURO STOXX 50 à partir des données Yahoo Finance :
- Fenêtres **glissantes** (tous les mois, sur 60 mois)
- Fenêtres **non chevauchantes** (blocs de 60 mois)
- Affiche **CAGR annualisé** et **rendement cumulé** sur 5 ans
- Option pour tracer la courbe du CAGR 5 ans glissant

## Prérequis
- Python ≥ 3.9
- `yfinance`, `pandas`, `matplotlib`

```bash
pip install yfinance pandas matplotlib
```

## Utilisation
```bash
python eurostoxx5y.py --ticker "^STOXX50E" --plot
```

**Options :**
- `--ticker` : `^STOXX50E` (indice prix, sans dividendes) par défaut.  
  Pour dividendes réinvestis (si dispo), utilisez `^SX5T`.
- `--plot` : affiche un graphique du CAGR 5 ans glissant.

## Exemple de sortie
```
Indice: ^STOXX50E
Période disponible: 2007-03-31 → 2025-08-31
Nombre de fenêtres glissantes 5 ans: 162
— Rendements 5 ans (fenêtres glissantes) —
  • CAGR moyen (annualisé): 2.87%
  • CAGR médian (annualisé): 4.27%
  • Rendement cumulé moyen sur 5 ans: 18.15%
  • Rendement cumulé médian sur 5 ans: 23.26%

Fenêtres non chevauchantes (tous les 60 mois) : 3 bloc(s)
  • CAGR moyen (annualisé): -0.19%
  • CAGR médian (annualisé): 2.20%
  • Rendement cumulé moyen sur 5 ans: 4.01%
  • Rendement cumulé médian sur 5 ans: 11.47%
```

## Notes
- Les résultats pour `^STOXX50E` **n’incluent pas** les dividendes.
- Les calculs sont faits **depuis la première date disponible** sur Yahoo Finance.
