# utils/alerts.py — simple SMTP alerts for FluxTwin
from __future__ import annotations
import smtplib
from email.message import EmailMessage
from typing import Optional

def send_email_smtp(
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str,
    from_name: str = "FluxTwin Alerts",
    from_email: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Απλή αποστολή email μέσω SMTP (TLS).
    Επιστρέφει (ok, message).
    """
    from_email = from_email or smtp_user

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{from_email}>"
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, f"error: {e}"


def format_spike_alert(project: str, last_value_kwh: float, baseline_kwh: float, threshold_pct: float) -> str:
    return (
        f"[{project}] Spike alert\n"
        f"- Last reading: {last_value_kwh:.2f} kWh\n"
        f"- Baseline (avg): {baseline_kwh:.2f} kWh\n"
        f"- Threshold: +{threshold_pct:.0f}% over baseline\n\n"
        f"Suggested actions:\n"
        f"• Check unexpected loads / equipment left ON\n"
        f"• Inspect HVAC setpoints and schedules\n"
        f"• Verify PV/backup systems and meter readings\n"
    )


def format_forecast_alert(project: str, horizon_days: int, est_cost_no: float, est_cost_after: float, budget_eur: float) -> str:
    return (
        f"[{project}] Forecast risk alert\n"
        f"- Horizon: {horizon_days} days\n"
        f"- Estimated cost (no action): {est_cost_no:.2f} €\n"
        f"- Estimated cost (after actions): {est_cost_after:.2f} €\n"
        f"- Budget: {budget_eur:.2f} €\n\n"
        f"Suggested actions:\n"
        f"• Shift non-critical loads to off-peak hours\n"
        f"• Tighten HVAC setpoints during peak tariff windows\n"
        f"• Review automation schedules & occupancy profiles\n"
    )
