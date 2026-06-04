$ErrorActionPreference = "Stop"

python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
