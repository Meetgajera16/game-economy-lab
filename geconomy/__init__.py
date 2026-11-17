"""game-economy-lab: simulated collectible-pack game economy.

Modules
-------
packs        pack odds, expected value, payout distributions, rarity calibration, set completion
simulate     acquisition channels + player purchase behaviour with known ground truth
sql_metrics  SQL (SQLite) KPI layer cross-checked against pandas
ltv          BG/NBD + Gamma-Gamma customer lifetime value
retention    Kaplan-Meier / log-rank / Cox analysis of lapse
experiment   A/B analysis, power, CUPED, sequential looks, uplift targeting
acquisition  channel response curves and LTV-aware budget allocation
matchmaking  Elo / Glicko / Bradley-Terry ratings and pairing policies

All data are SIMULATED. Every parameter is an assumption stated in the code.
"""
