import sys
import json
import re
from pathlib import Path

# Setup path mappings for config/routing compatibility
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router

SYSTEM_INTERVIEW = """You are a senior technical recruiter and professional interview coach.
You generate highly specific, context-aware interview questions, suggested answers, and prep tools.
Return ONLY valid JSON, no markdown fences, no conversational preamble, no postamble."""

INTERVIEW_PROMPT = """
Based on the candidate's resume and the job description, generate an interview prep kit in valid JSON format.

CANDIDATE'S RESUME EXCERPT:
{resume_excerpt}

TARGET JOB DETAILS:
Title: {job_title}
Company: {company}
Domain: {domain}
Required Skills: {skills}

You MUST return ONLY a JSON object with this exact structure:
{{
  "behavioral": [
    {{
      "question": "Behavioral question using STAR format (e.g. Tell me about a time...)",
      "why_asked": "Recruiter's underlying rationale",
      "model_answer": "Model STAR framework response (Situation, Task, Action, Result) based on candidate's real accomplishments"
    }}
  ], // EXACTLY 5 behavioral questions
  
  "technical": [
    {{
      "question": "Technical question targeting required skills",
      "topic": "Skill or tool targeted",
      "expected_answer": "Detailed technical points candidate should speak to"
    }}
  ], // EXACTLY 5 technical questions
  
  "company": [
    "Smart, highly custom question candidate should ask the interviewer about company or role"
  ], // EXACTLY 3 custom questions
  
  "red_flags": [
    "Topic candidate must prepare carefully (e.g. gap, technology transition, or seniority difference) and how to address it"
  ] // EXACTLY 2-3 topics
}}

Job Description Metadata:
{jd_json}

Ensure the output is 100% valid JSON and does not contain markdown code fences.
"""

def generate_prep(resume_text: str, jd_dict: dict) -> dict:
    """
    Generates a role-specific, comprehensive interview preparation kit.
    """
    # extract first 1500 chars of resume for context speed
    resume_excerpt = resume_text[:1500]
    
    prompt = INTERVIEW_PROMPT.format(
        resume_excerpt=resume_excerpt,
        job_title=jd_dict.get("job_title", "Software Engineer"),
        company=jd_dict.get("company", "Target Company"),
        domain=jd_dict.get("domain", "Technology"),
        skills=", ".join(jd_dict.get("required_skills", [])),
        jd_json=json.dumps(jd_dict)
    )
    
    raw_response = llm_router.generate(prompt, force_local=True)
    
    # Extract JSON robustly from conversational preambles/postambles
    json_str = ""
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        match_generic = re.search(r"```\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
        if match_generic:
            json_str = match_generic.group(1)
        else:
            start_idx = raw_response.find("{")
            end_idx = raw_response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = raw_response[start_idx:end_idx+1]
            else:
                json_str = raw_response
                
    try:
        return json.loads(json_str.strip())
    except Exception as e:
        print(f"[Interview Prep] Failed to parse JSON from LLM: {e}. Raw was:\n{raw_response}")
        # Safe fallback structure
        return {
            "behavioral": [
                {
                    "question": "Tell me about a time you solved a difficult technical problem.",
                    "why_asked": "To understand your problem-solving process.",
                    "model_answer": "S: When working on optimizations; T: I needed to reduce load times; A: I implemented parallel processing; R: Resulting in a 40% improvement."
                }
            ],
            "technical": [
                {
                    "question": "How do you handle dependency injection and state management in Python?",
                    "topic": "Python Core",
                    "expected_answer": "Explain clean architecture, modular classes, and decouple utilities."
                }
            ],
            "company": [
                "What does the engineering culture look like regarding learning and deployment sprints?"
            ],
            "red_flags": [
                "Be ready to elaborate on any gap in your resume using positive learning outcomes."
            ]
        }

def generate_deep_project_drill(resume_text: str, project_name: str, 
                                 model_source: str) -> dict:
    """
    Generates hyper-specific technical Q&A for a given project.
    ONLY runs when model_source == 'ollama'. Returns empty dict otherwise.
    """
    if model_source != "ollama":
        return {}
    
    base_prompt = f"""
You are a senior staff engineer conducting a brutal technical interview.
The candidate claims to have built this project: {project_name}
Here is their full resume context: {resume_text}

Generate a deep technical drill in STRICT JSON format with these exact keys:
{{
  "architecture": [
    {{"q": "question", "a": "200-300 word answer explaining the real technical reason"}}
    ... 5 items
  ],
  "internals": [
    {{"q": "question about exact tool mechanics", "a": "150-200 word precise answer"}}
    ... 8 items  
  ],
  "failure_modes": [
    {{"q": "stress test / tradeoff question", "a": "answer", "follow_up": "what senior asks next"}}
    ... 5 items
  ],
  "rapid_fire": [
    {{"q": "only-a-builder-knows question", "a": "1-2 line answer"}}
    ... 10 items
  ],
  "concept_gaps": [
    {{"term": "advanced term from resume", "q": "explain like senior question", "a": "150 word answer"}}
    ... 5-10 items
  ]
}}

Rules:
- Architecture Qs: ask WHY design choices were made, not WHAT was built
- Internals Qs: ask about exact mechanics e.g. "what does MLflow write to disk?", 
  "what is the input tensor shape to YOLO?", "how does GroundedSAM combine its two models?"
- Failure mode Qs: edge cases, what breaks, what was the tradeoff chosen
- Rapid fire: things only the coder knows — HTTP methods, specific parameters, exact metrics
- Concept gaps: detect terms like "agentic workflows", "distributed microservices", 
  "CI/CD pipeline", "LLM orchestration", "semantic similarity", "ensemble model" in 
  the resume and generate explain-from-first-principles questions for each
- Return ONLY valid JSON. No preamble. No markdown code fences. No explanation.
"""
    
    raw = llm_router.generate(base_prompt, force_local=True)

    return _parse_drill_json(raw)


def _extract_json_object(text: str) -> str:
    """
    Walks `text` character-by-character to extract the outermost {...} JSON object.
    Much more reliable than greedy regex on deeply nested structures.
    """
    depth = 0
    in_string = False
    escape_next = False
    start = None

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return ""


def _parse_drill_json(raw: str) -> dict:
    """
    Multi-strategy JSON parser for Ollama deep drill output.
    Tries strategies in order of reliability.
    """
    import json

    # Strategy 1 ── strip markdown fences then try direct parse
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence line
        cleaned = cleaned[cleaned.find("\n") + 1 :]
    if "```" in cleaned:
        cleaned = cleaned[: cleaned.rfind("```")]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2 ── balanced-brace extraction
    extracted = _extract_json_object(cleaned)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Strategy 3 ── remove trailing commas (common Ollama bug) then parse
    repaired = re.sub(r",\s*([}\]])", r"\1", extracted or cleaned)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Strategy 4 ── section-by-section partial parse (never fail completely)
    sections = {}
    for section in ["architecture", "internals", "failure_modes", "rapid_fire", "concept_gaps"]:
        pattern = rf'"{section}"\s*:\s*(\[.*?\])(?=\s*[,}}])'
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            try:
                sections[section] = json.loads(m.group(1))
            except Exception:
                pass

    if sections:
        return sections

    # Absolute fallback — surface raw for debugging
    return {"error": "Drill generation failed. Try again.", "raw": raw[:800]}

