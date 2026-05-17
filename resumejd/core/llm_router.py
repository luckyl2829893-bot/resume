import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Use the new google-genai SDK (google.generativeai is deprecated)
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _USE_NEW_SDK = True
except ImportError:
    # Fallback to old SDK if new one not installed
    import google.generativeai as _genai_old
    _USE_NEW_SDK = False

# Append parent dir to path to ensure standard imports work seamlessly
sys.path.append(str(Path(__file__).parent.parent))
from config import OLLAMA_URL, OLLAMA_MODEL, GEMINI_MODEL

# Load environmental variables from local .env relative to the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── Module-level tracking of which model was actually used ─────────────────────
_last_model_used: str = "unknown"
_last_model_label: str = "Not run yet"


def get_last_model_used() -> dict:
    """Returns the last model name and display label used by generate()."""
    return {"model": _last_model_used, "label": _last_model_label}


def generate(prompt: str, force_local: bool = False, force_gemini: bool = False) -> str:
    """
    Routes prompt to the appropriate LLM.

    Priority logic:
      force_gemini=True  → always use Gemini (error if no key)
      force_local=True   → always use Ollama
      neither            → try Gemini first, fall back to Ollama on any error

    Sets _last_model_used so the UI can display which model ran.
    Returns the string text output.
    """
    global _last_model_used, _last_model_label

    gemini_key = os.getenv("GEMINI_API_KEY")

    if force_gemini and gemini_key:
        result = _generate_gemini(prompt, gemini_key)
        if result is not None:
            return result
        _last_model_used = "gemini_failed"
        _last_model_label = "Gemini FAILED — check key/quota"
        return "ERROR: Gemini request failed. Check your API key and quota."

    if force_local or not gemini_key:
        _last_model_used = "ollama"
        _last_model_label = f"Ollama ({OLLAMA_MODEL})"
        return _generate_ollama(prompt)

    # Auto mode: try Gemini, fall back to Ollama
    result = _generate_gemini(prompt, gemini_key)
    if result is not None:
        return result

    print(f"[LLM Router] Gemini failed. Falling back to local Ollama ({OLLAMA_MODEL})...")
    _last_model_used = "ollama_fallback"
    _last_model_label = f"Ollama fallback ({OLLAMA_MODEL})"
    return _generate_ollama(prompt)


def _generate_gemini(prompt: str, api_key: str) -> str | None:
    """Calls Gemini using the new google-genai SDK. Returns text on success, None on failure."""
    global _last_model_used, _last_model_label
    try:
        if _USE_NEW_SDK:
            client = _genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(temperature=0.3),
            )
            _last_model_used = "gemini"
            _last_model_label = f"Gemini ({GEMINI_MODEL})"
            return response.text
        else:
            _genai_old.configure(api_key=api_key)
            model = _genai_old.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            _last_model_used = "gemini"
            _last_model_label = f"Gemini ({GEMINI_MODEL}) [legacy SDK]"
            return response.text
    except Exception as e:
        print(f"[LLM Router] Gemini error: {e}")
        return None


def _generate_ollama(prompt: str) -> str:
    """Executes local inference via Ollama."""
    global _last_model_used, _last_model_label
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        _last_model_used = "ollama"
        _last_model_label = f"Ollama ({OLLAMA_MODEL})"
        return response.json().get("response", "")
    except Exception as e:
        print(f"[LLM Router] Ollama request failed: {e}")
        _last_model_used = "ollama_error"
        _last_model_label = "Ollama ERROR"
        return f"ERROR: Local Ollama is offline or model '{OLLAMA_MODEL}' is missing. Run 'ollama serve'."


def check_ollama_status() -> dict:
    """Pings local Ollama to determine availability and list active models."""
    try:
        base_url = OLLAMA_URL.rsplit("/", 2)[0]
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return {
                "running": True,
                "models": models,
                "target_model_installed": any(OLLAMA_MODEL in m for m in models),
            }
    except Exception:
        pass
    return {"running": False, "models": [], "target_model_installed": False}
