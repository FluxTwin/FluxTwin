import numpy as np
import pandas as pd

# Απλές default τιμές για Κύπρο (μέση οριζόντια ηλιακή ακτινοβολία ~5.5 kWh/m2/day)
# και απόδοση συστήματος ~80% (losses+inverter).
DEFAULT_GHI = 5.5    # kWh/m2/day
DEFAULT_PR  = 0.80   # performance ratio

def pv_daily_kwh(kw_p: float, ghi: float = DEFAULT_GHI, pr: float = DEFAULT_PR) -> float:
    """Πολύ απλός εκτιμητής ημερήσιας παραγωγής σε kWh."""
    return kw_p * ghi * pr

def simulate_roi(
    kw_p: float,
    price_eur_per_kwh: float,
    capex_eur: float,
    self_consumption_ratio: float = 0.8,
    days: int = 365
) -> dict:
    """Εκτιμά παραγωγή, έσοδα/εξοικονόμηση και payback (χωρίς API)."""
    daily = pv_daily_kwh(kw_p)
    annual_kwh = daily * 365
    used_onsite = annual_kwh * self_consumption_ratio
    savings_eur = used_onsite * price_eur_per_kwh
    simple_payback_years = capex_eur / max(savings_eur, 1e-6)
    return {
        "daily_kwh": daily,
        "annual_kwh": annual_kwh,
        "used_onsite_kwh": used_onsite,
        "annual_savings_eur": savings_eur,
        "payback_years": simple_payback_years,
        "assumptions": {"ghi": DEFAULT_GHI, "pr": DEFAULT_PR, "self_consumption": self_consumption_ratio}
    }
