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


def generate(prompt: str, force_local: bool = False,
             force_gemini: bool = False, api_key: str = None) -> str:
    """
    Unified LLM router.
    Priority order when neither force flag is set:
      1. Gemini (fast, use for short tasks)
      2. Ollama fallback on 429 / missing key / any exception
    api_key overrides .env GEMINI_API_KEY if provided.
    """
    global _last_model_used, _last_model_label

    # Force local → skip Gemini entirely
    if force_local and not force_gemini:
        _last_model_used = "ollama"
        _last_model_label = f"Ollama ({OLLAMA_MODEL})"
        return _call_ollama(prompt)

    # Try Gemini first
    if not force_local:
        try:
            res = _call_gemini(prompt, api_key=api_key)
            _last_model_used = "gemini"
            _last_model_label = f"Gemini ({GEMINI_MODEL})"
            return res
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit  = "429" in err_str or "quota" in err_str
            is_key_missing = "api_key" in err_str or "invalid" in err_str or "no gemini api key" in err_str
            if is_rate_limit or is_key_missing or not force_gemini:
                # Fall back to Ollama
                try:
                    res = _call_ollama(prompt)
                    _last_model_used = "ollama_fallback"
                    _last_model_label = f"Ollama fallback ({OLLAMA_MODEL})"
                    return res
                except Exception as ollama_err:
                    _last_model_used = "all_failed"
                    _last_model_label = "All models failed"
                    raise RuntimeError(
                        f"Both Gemini and Ollama failed.\n"
                        f"Gemini: {e}\nOllama: {ollama_err}"
                    )
            _last_model_used = "gemini_failed"
            _last_model_label = "Gemini FAILED"
            raise

    # Auto mode fallback: if force_local was somehow False but we ended up here
    _last_model_used = "ollama_fallback"
    _last_model_label = f"Ollama fallback ({OLLAMA_MODEL})"
    return _call_ollama(prompt)


def _call_gemini(prompt: str, api_key: str = None) -> str:
    """Calls Gemini using google-genai or legacy google.generativeai SDK."""
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("No Gemini API key available")

    if _USE_NEW_SDK:
        client = _genai.Client(api_key=key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(temperature=0.3),
        )
        return response.text
    else:
        _genai_old.configure(api_key=key)
        model = _genai_old.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text


def _call_ollama(prompt: str) -> str:
    """Executes local inference via Ollama."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 3072},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        if not text or not text.strip():
            return f"ERROR: Ollama returned an empty response. The model '{OLLAMA_MODEL}' may still be loading."
        return text
    except requests.exceptions.ConnectionError:
        return f"ERROR: Ollama is not running. Start it with 'ollama serve' and ensure model '{OLLAMA_MODEL}' is pulled."
    except requests.exceptions.Timeout:
        return f"ERROR: Ollama timed out (300s)."
    except Exception as e:
        return f"ERROR: Local Ollama failed: {e}."


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
