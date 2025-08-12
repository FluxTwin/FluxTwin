# utils/advisor_ai.py
from __future__ import annotations
import numpy as np
import pandas as pd

def smart_advice(profile: dict, kpis: dict | None = None, forecast_df: pd.DataFrame | None = None) -> dict:
    """
    Επιστρέφει λεξικό με:
      - tips: λίστα από actionable προτάσεις (strings)
      - expected_savings_pct: εκτιμώμενο ποσοστό εξοικονόμησης (0–1)

    Το ποσοστό είναι συντηρητικό και προσαρμόζεται με βάση το προφίλ
    και (αν υπάρχει) τη μεταβλητότητα του forecast.
    """
    usage = (profile.get("type") or "office").lower()
    has_pv = bool(profile.get("has_pv", True))

    # Βασικές ζώνες (συντηρητικές) ανά χρήση
    base = {
        "household": (0.07, 0.12),
        "office":    (0.10, 0.18),
        "hotel":     (0.08, 0.15),
        "factory":   (0.05, 0.12),
    }
    lo, hi = base.get(usage, (0.08, 0.14))

    # Μικρή προσαύξηση αν υπάρχει μεγάλη μεταβλητότητα στο forecast
    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df:
        vol = float(np.std(forecast_df["forecast_kwh"])) if len(forecast_df) > 1 else 0.0
        bump = min(0.03, vol / 500.0)  # πολύ συντηρητικό
        hi = min(hi + bump, 0.22)

    # Αν υπάρχει PV, λίγο υψηλότερη δυνητική εξοικονόμηση
    if has_pv:
        lo += 0.01
        hi += 0.01

    expected = round((lo + hi) / 2, 3)

    # Actionable tips ανά προφίλ
    tips: list[str] = []
    if usage in ("office", "hotel"):
        tips += [
            "Shift non-critical loads to off-peak hours (servers backup, laundry).",
            "HVAC setpoints: +1 °C in cooling / −1 °C in heating outside peak hours.",
            "Install occupancy sensors & daylight dimming in corridors/meeting rooms.",
            "Weekly schedule review: turn off AHUs/VRF per zone after 18:00.",
        ]
    if usage == "hotel":
        tips += [
            "Hot-water recirculation: timer + temperature band control.",
            "Pool filtration to off-peak; cover pool at night to reduce losses.",
        ]
    if usage == "factory":
        tips += [
            "Compressors: fix leaks, cascaded pressure setpoints, regular maintenance.",
            "Stagger high-load machines (15-min ramps) to flatten peaks.",
        ]
    if usage == "household":
        tips += [
            "Time-shift dishwasher/washing machine to off-peak.",
            "Smart plugs for standby killers (TV, set-top boxes, chargers).",
        ]

    # PV-related
    if has_pv:
        tips += [
            "Run heat-pumps/boilers for pre-heating water during PV peak (11:00–15:00).",
            "Consider small battery (3–5 kWh) to shave evening peaks.",
        ]

    # Κοινές πρακτικές
    tips += [
        "Create a simple weekly energy checklist; assign an owner per system.",
        "Enable automated alerts when hourly usage >120% of baseline.",
    ]

    return {"tips": tips, "expected_savings_pct": expected}
