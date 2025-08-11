# FluxTwin — Starter (MVP)

Πρώτη έκδοση της πλατφόρμας **FluxTwin**: ανέβασε ενεργειακά δεδομένα, δες ανάλυση, πάρε AI σύνοψη (demo) και εξήγαγε PDF.

## 🚀 Γρήγορη εκκίνηση (τοπικά)

1) Δημιούργησε ένα virtual environment (προαιρετικό) και εγκατέστησε τα dependencies:
```bash
pip install -r requirements.txt
```

2) (Προαιρετικό) Αντιγραφή `.env.example` σε `.env` και βάλε το `OPENAI_API_KEY`:
```bash
cp .env.example .env
# EDIT .env
```

3) Τρέξε την εφαρμογή:
```bash
streamlit run app.py
```

4) Άνοιξε τον σύνδεσμο που θα εμφανίσει το Streamlit (συνήθως http://localhost:8501).

## 🧪 Δοκιμή
- Από το sidebar, κατέβασε το `sample_data.csv` και φόρτωσέ το.
- Δες δείκτες, γράφημα, ημερήσια σύνοψη.
- Πάτα **Δημιουργία PDF** για να κατεβάσεις αναφορά.

## 🧠 AI Σύνοψη
- Αν προσθέσεις `OPENAI_API_KEY` στο `.env`, στην παραγωγή η σύνοψη θα προέρχεται από μοντέλο AI.
- Σε αυτό το starter, δείχνουμε demo κείμενο (δεν γίνονται κλήσεις σε εξωτερικά APIs εδώ).

## ☁️ Ανάπτυξη σε Streamlit Cloud
1) Ανέβασε αυτό το repo στο GitHub.
2) Στο [Streamlit Cloud](https://streamlit.io/cloud) κάνε Deploy το repo.
3) Πρόσθεσε στα **Secrets** το `OPENAI_API_KEY` και οποιαδήποτε άλλα env vars.

## 🗺️ Οδικός Χάρτης (επόμενα βήματα)
- Forecasting (πρόβλεψη) με κλασικά μοντέλα + AI βοηθό.
- Πολυ-πηγές: ρεύμα, νερό, καύσιμα, φυσικό αέριο.
- Benchmarks και ROI calculators.
- White-label και Enterprise features (RBAC, audit logs, SSO).
- API για ERP/CRM και υποστήριξη IoT μετρητών.

## 📄 Άδεια
MIT License.