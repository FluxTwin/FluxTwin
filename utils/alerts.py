# utils/alerts.py
import smtplib
from email.mime.text import MIMEText

def smtp_ready(secrets) -> bool:
    try:
        return all(k in secrets for k in ["SMTP_SERVER", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"])
    except Exception:
        return False

def send_email_alert(secrets, subject: str, body: str, to_addr: str | None = None) -> tuple[bool, str]:
    if not smtp_ready(secrets):
        return False, "SMTP secrets missing"
    to_addr = to_addr or secrets.get("ALERT_TO", "")
    if not to_addr:
        return False, "No ALERT_TO provided"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = secrets["SMTP_USER"]
    msg["To"] = to_addr

    try:
        server = smtplib.SMTP(secrets["SMTP_SERVER"], int(secrets["SMTP_PORT"]))
        server.starttls()
        server.login(secrets["SMTP_USER"], secrets["SMTP_PASSWORD"])
        server.sendmail(secrets["SMTP_USER"], [to_addr], msg.as_string())
        server.quit()
        return True, "Sent"
    except Exception as e:
        return False, f"SMTP error: {e}"
