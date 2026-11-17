"""Unit economics shared by the experiment and acquisition modules.

ASSUMPTION (stated, not estimated): the platform's cost of delivering a card is
CARD_COST_FRACTION of its market value. Together with the pack module's calibrated
return-to-player this fixes the contribution margin on revenue and the cost of a
free bonus pack. Change these two constants and everything downstream updates.
"""
from __future__ import annotations

from . import packs

TARGET_RTP = 0.85
CARD_COST_FRACTION = 0.35
ANNUAL_DISCOUNT = 0.10


def unit_economics() -> dict:
    spec, scale = packs.calibrate_rarity(packs.default_pack(), TARGET_RTP)
    ev = packs.pack_ev(spec)
    return {
        "pack_price": spec.price,
        "rtp": packs.rtp(spec),
        "epic_plus_odds_scale": scale,
        "pack_ev_market_value": ev,
        "cogs_per_pack": ev * CARD_COST_FRACTION,
        "margin_rate": 1.0 - packs.rtp(spec) * CARD_COST_FRACTION,   # contribution margin / revenue
        "bonus_pack_cost": ev * CARD_COST_FRACTION,                  # cost of one free pack
    }
