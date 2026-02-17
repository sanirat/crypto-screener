# Deploy to Render (Streamlit)

This repo is ready for Render using Docker.

## Steps
1. Create a new GitHub repo and add these files:
   - app.py
   - requirements.txt
   - Dockerfile
   - render.yaml (optional)

2. In Render:
   - New + → **Web Service**
   - Connect your GitHub repo
   - Environment: **Docker**
   - Plan: Free (or higher)
   - Create Web Service

Render will build the Docker image and start Streamlit.

## Notes
- CoinGecko rate limits: keep scan pages modest, or upgrade Render plan for more stable perf.
- Streamlit caching is enabled to reduce API calls.
