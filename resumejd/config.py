import os

# ── DATABASE SETTINGS ──
# Resolve parent DB path dynamically to avoid path issues
DB_PATH = "resumejd/db/applications.db"

# ── OLLAMA LOCAL SETTINGS ──
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# ── GEMINI SETTINGS ──
GEMINI_MODEL = "gemini-2.5-flash"

# ── SCORING THRESHOLDS ──
ATS_THRESHOLD = 75
