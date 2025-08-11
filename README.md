# FluxTwin — Energy Analytics (MVP)

English-first MVP for FluxTwin:
- Upload CSV with `timestamp, consumption_kwh`
- KPIs + chart + daily table
- AI summary (optional, requires OpenAI API key)
- Daily **7–30 day forecast** (Holt–Winters / naive mean)
- Export **PDF** with KPIs, AI summary, consumption chart, daily table, forecast chart/table

## Quick start (local)
```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY (optional)
streamlit run app.py
```

## Deploy to Streamlit Cloud
- Connect your GitHub repo
- Set **Secrets**:
```
OPENAI_API_KEY = "sk-...optional..."
FLUXTWIN_APP_NAME = "FluxTwin"
```
- Deploy `app.py`

## Roadmap (short)
- Multi-source (electricity, water, fuels)
- Benchmarks & anomaly detection
- White-label + Org/Users + SSO
- API & IoT connectors