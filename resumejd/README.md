# 📄 resumejd v2 — Privacy-First AI Resume Workspace

A premium, local, and privacy-first AI Resume Optimizer and Job Application Workspace. Tailor your resume to any job description using the **Google X-Y-Z formula**, generate customized assets, drill projects, and track applications locally.

---

## 🚀 Key Features

*   **🏠 Resume Tailorer & Matcher**
    *   Optimizes bullet points with metrics using the **Google X-Y-Z formula**.
    *   **Layout Fingerprinting:** PyMuPDF layout analysis extracts and preserves original font sizes, accents, and section order (retaining original document structures perfectly).
    *   Side-by-side ATS match metrics using semantic **Sentence-Transformers** CPU similarity.
*   **📨 Cover Letter Generator**
    *   Generates a high-converting, 3-paragraph tailored cover letter targeting role requirements.
*   **🎤 Interview Prep Deep Drill**
    *   Provides technical drills and failure-mode analysis on specific resume projects using Ollama.
*   **📊 Job Tracker & Agent**
    *   Local SQLite database to track job pipelines.
    *   Autonomous Web Agent (Playwright-based) to dry-run or auto-apply to roles safely.
*   **🧪 Collapsible Settings & MLOps Log**
    *   Manage local `.env` keys and Ollama status.
    *   Audit telemetry runs showing historical score improvements and engines used.

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **LLM Orchestrator:** Google Gemini (Generative AI client) + local offline **Ollama** (Llama 3.1 8b fallback)
*   **Semantic Scoring:** `all-MiniLM-L6-v2` Sentence-Transformers + Cosine Similarity
*   **Layout Analysis & Generation:** PyMuPDF + python-docx + fpdf2
*   **Automation Engine:** Playwright
*   **Database:** Local SQLite

---

## 💻 Quick Start

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Or configure the key directly in the sidebar expander inside the app).*

### 3. Run Streamlit
Launch the workspace:
```bash
streamlit run app.py
```
