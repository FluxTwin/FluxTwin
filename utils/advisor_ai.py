import os
from . import forecasting as fc

def smart_advice(profile: dict, kpis: dict, forecast_df):
    """
    Rule+AI-ready συμβουλές.
    Αν υπάρχει OPENAI_API_KEY στο περιβάλλον, μπορείς να καλέσεις μοντέλο αργότερα.
    Τώρα δίνουμε στοχευμένες προτάσεις με βάση τύπο χρήστη & forecast.
    """
    typ = (profile.get("type") or "general").lower()
    price = float(profile.get("price_eur_per_kwh", 0.25))
    horizon_kwh = float(forecast_df["forecast_kwh"].sum()) if forecast_df is not None else 0.0
    est_cost = horizon_kwh * price

    tips = []
    if typ in ["hotel", "hospitality", "office"]:
        tips += [
            "Shift HVAC setpoints by +1°C during peak hours to cut 5–10% cooling load.",
            "Automate lighting with occupancy sensors in low-traffic corridors.",
        ]
    if typ in ["factory","industrial"]:
        tips += [
            "Move non-critical processes to off-peak hours; aim to shave 10–15% peak demand.",
            "Check compressed-air leaks; typical savings 5–8%.",
        ]
    if profile.get("has_pv", False):
        tips += ["Schedule high-load appliances when PV output is highest (11:00–15:00)."]

    if est_cost > 0:
        tips.insert(0, f"Projected energy cost next window: ~€{est_cost:,.2f}. Target a 7–15% cut with the actions below.")
    if not tips:
        tips = ["Reduce simultaneous use of high-load devices during 11:00–18:00.", "Switch to LED and timer controls."]
    return tips
