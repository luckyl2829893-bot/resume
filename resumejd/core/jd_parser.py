import sys
import json
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Setup path imports for config/routing compatibility
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router

SYSTEM_JD = """You are an expert ATS system analyst and recruitment specialist.
Extract ALL key information from job descriptions accurately and completely.
Return ONLY valid JSON, no markdown, no preamble, no commentary."""

JD_EXTRACTION_PROMPT = """
Analyze this job description and extract the following. Return ONLY valid JSON:

{{
  "job_title": "exact title or null",
  "company": "company name or null",
  "required_skills": ["skill1", "skill2"],        // MUST-HAVE skills
  "preferred_skills": ["skill1", "skill2"],       // Nice-to-have skills
  "tools_and_technologies": ["Python", "AWS"],    // Specific tools/tech mentioned
  "responsibilities": ["bullet 1", "bullet 2"],   // Key job duties
  "qualifications": ["degree requirement", "cert"],
  "keywords_critical": ["ATS-critical keywords — exact phrases from JD"],
  "keywords_secondary": ["supporting keywords"],
  "domain": "e.g. Machine Learning, Web Dev, Finance, Marketing",
  "seniority": "fresher/junior/mid/senior/lead/manager",
  "culture_keywords": ["innovative", "fast-paced"]
}}

Job Description:
{jd_text}
"""

def parse_jd(text: str) -> dict:
    """
    Analyzes job description raw text via LLM router and extracts structured JSON.
    """
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=text)
    raw_response = llm_router.generate(prompt)
    
    # Extract JSON robustly from conversational preambles/postambles
    json_str = ""
    # 1. Try to find content within ```json ... ``` fences
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        # 2. Try to find content within standard ``` ... ``` fences
        match_generic = re.search(r"```\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
        if match_generic:
            json_str = match_generic.group(1)
        else:
            # 3. Find content between first '{' and last '}'
            start_idx = raw_response.find("{")
            end_idx = raw_response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = raw_response[start_idx:end_idx+1]
            else:
                json_str = raw_response
                
    try:
        return json.loads(json_str.strip())
    except Exception as e:
        print(f"[JD Parser] Failed to parse JSON from LLM: {e}. Raw was:\n{raw_response}")
        # Safety fallback schema dictionary in case LLM outputs malformed format
        return {
            "job_title": "Unknown Role",
            "company": "Unknown Company",
            "required_skills": [],
            "preferred_skills": [],
            "tools_and_technologies": [],
            "responsibilities": [],
            "qualifications": [],
            "keywords_critical": [],
            "keywords_secondary": [],
            "domain": "General",
            "seniority": "mid",
            "culture_keywords": []
        }

def scrape_jd(url: str) -> str:
    """
    Scrapes a job posting web page, strips scripts/styles/navigation/footers,
    and returns clean, structured text for JD extraction.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove layout boilerplate
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            element.extract()
            
        # Get raw content text, clean spacing
        raw_text = soup.get_text(separator="\n")
        lines = (line.strip() for line in raw_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
        
        return cleaned_text
    except Exception as e:
        raise RuntimeError(f"Failed to scrape job description: {e}")
