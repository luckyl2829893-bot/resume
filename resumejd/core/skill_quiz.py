"""
core/skill_quiz.py

Resume-grounded skill quiz generator and test runner.
Anti-hallucination strategy:
- Extract skills/tools/projects ONLY from the resume text.
- Questions include the skill name + project/context from the resume as grounding.
- Returns scenario-based MCQs.
- Free Gemini tier cap: 10 questions.
- 11+ questions: routes to Ollama if available, else caps at 10 and flags needs_ollama.
"""

import json
import re
import sys
from pathlib import Path

# Setup path mappings for config/routing compatibility
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router

SYSTEM_EXTRACTOR = """Extract skills and projects from resumes accurately.
Return ONLY valid JSON. Never invent skills not in the resume."""

SYSTEM_QUIZ = """You are a senior technical interviewer generating deep, practical quiz questions.
Rules:
1. Questions must be answerable ONLY by someone who actually worked with that skill.
2. NO surface-level definition questions ("What is X?").
3. Include edge cases, common mistakes, debugging scenarios, architectural decisions.
4. For project-based questions: reference the specific project context from the resume.
5. Every question must have exactly 4 options (A/B/C/D) with ONE correct answer.
6. Include a one-line explanation for the correct answer.
7. Return ONLY valid JSON — no markdown, no preamble."""

GEMINI_CAP = 10  # safe per-call limit for Gemini free tier
def extract_resume_skills(resume_text: str, router_instance=None, api_key: str = None) -> dict:
    """
    Parse resume and extract structured skill/project/tool profile.
    Uses an exhaustive prompt to catch every technical term in the resume,
    including ones mentioned only in bullet point context.
    """
    prompt = f"""
Read this resume carefully — every single line, every bullet point, every project description.
Extract ALL technical skills, tools, frameworks, libraries, platforms, and technologies you can find.
Do not skip anything mentioned in passing or in context sentences.
 
Return ONLY JSON in this exact format (no markdown, no preamble):
 
{{
  "programming_languages": ["Python", "JavaScript", "C++"],
  "frameworks_libraries": ["FastAPI", "React", "YOLOv11", "Transformers"],
  "tools_platforms": ["Git", "Docker", "AWS", "VS Code", "Postman"],
  "databases": ["PostgreSQL", "MongoDB", "SQLite", "Redis"],
  "concepts": ["REST API", "Machine Learning", "Computer Vision", "CI/CD"],
  "projects": [
    {{
      "name": "Project Name",
      "tech_stack": ["Python", "FastAPI", "MongoDB"],
      "description": "one line of what it does",
      "key_features": ["feature 1", "feature 2"]
    }}
  ],
  "internships_jobs": [
    {{
      "role": "ML Engineer Intern",
      "company": "Company Name",
      "tech_used": ["Python", "TensorFlow", "Docker"],
      "key_work": "brief description of what was built or done"
    }}
  ],
  "domain_expertise": ["Computer Vision", "NLP", "Web Development"],
  "certifications": ["AWS Certified Developer", "TensorFlow Certificate"],
  "all_skills": ["Python", "FastAPI", "Docker", "MongoDB", "REST API", "YOLOv11"]
}}

CRITICAL RULES:
- The "all_skills" field must be a COMPLETE flat list of every distinct technical skill, tool,
  framework, library, platform, and concept you find ANYWHERE in the resume.
  This includes skills mentioned only once in a bullet point sentence.
- Do NOT skip tools mentioned in project descriptions, job duties, or achievement bullets.
- Include version numbers if mentioned (e.g. "YOLOv8", "Python 3.11", "Node.js 18").
- The goal is TOTAL COVERAGE — it is better to include too many than to miss any.

Resume text to analyze:
{resume_text}
"""
    if router_instance is None:
        raw_response = llm_router.generate(prompt, force_local=True, api_key=api_key)
    else:
        raw_response = router_instance.generate(prompt)

    # Detect router-level errors (Ollama offline, empty response, etc.)
    if not raw_response or not raw_response.strip() or raw_response.startswith("ERROR:"):
        raise RuntimeError(raw_response or "LLM returned an empty response. Check that Ollama is running.")

    # Clean markdown fences
    text = raw_response.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1 :]
    if "```" in text:
        text = text[: text.rfind("```")]
    text = text.strip()

    try:
        result = json.loads(text)
        # Ensure all_skills is always present and populated
        if "all_skills" not in result or not result["all_skills"]:
            all_s = set()
            for key in ["programming_languages", "frameworks_libraries", "tools_platforms",
                        "databases", "concepts", "domain_expertise"]:
                all_s.update(result.get(key, []))
            result["all_skills"] = sorted(all_s)
        return result
    except Exception as e:
        print(f"[Quiz Engine] Failed to parse skills JSON: {e}")
        # Fallback profile
        return {
            "programming_languages": ["Python"],
            "frameworks_libraries": ["Flask"],
            "tools_platforms": ["Git"],
            "databases": ["SQLite"],
            "concepts": ["Web Design"],
            "projects": [],
            "internships_jobs": [],
            "domain_expertise": [],
            "certifications": [],
            "all_skills": ["Python", "Flask", "Git", "SQLite"],
        }


def generate_quiz_batch(
    skills_profile: dict, focus_skills: list, num_questions: int, router_instance=None, api_key: str = None
) -> list:
    """
    Generate a batch of MCQ questions grounded in the candidate's resume.
    """
    projects_context = "\n".join(
        [
            f"- {p['name']}: {p['description']} | Stack: {', '.join(p['tech_stack'])}"
            for p in skills_profile.get("projects", [])
        ]
    )

    jobs_context = "\n".join(
        [
            f"- {j['role']} at {j['company']}: {j['key_work']} | Tech: {', '.join(j['tech_used'])}"
            for j in skills_profile.get("internships_jobs", [])
        ]
    )

    prompt = f"""
Generate exactly {num_questions} multiple-choice quiz questions to test a candidate
on the following skills extracted from their actual resume:

SKILLS TO TEST: {', '.join(focus_skills)}

CANDIDATE'S ACTUAL PROJECTS (use these for project-specific questions):
{projects_context if projects_context else "No projects listed"}

CANDIDATE'S WORK EXPERIENCE (use this for context):
{jobs_context if jobs_context else "No work experience listed"}

QUESTION TYPE DISTRIBUTION (enforce this mix):
- 40% Deep conceptual (internal mechanics, why X works, performance implications)
- 30% Project-specific (based on the candidate's actual listed projects above)
- 20% Debugging/troubleshooting (common errors, edge cases)
- 10% Best practices and architectural decisions

QUALITY RULES:
- Never ask "What is the definition of X?" — that is too basic.
- Never invent a project or technology not listed above.
- Make wrong answers plausible — not obviously incorrect.
- Each question must have exactly ONE unambiguously correct answer.

Return ONLY this JSON array with exactly {num_questions} items:
[
  {{
    "question": "full question text",
    "skill": "which skill this tests",
    "difficulty": "medium" | "hard",
    "type": "conceptual" | "project_specific" | "debugging" | "best_practice",
    "options": {{
      "A": "option text",
      "B": "option text",
      "C": "option text",
      "D": "option text"
    }},
    "correct": "A" | "B" | "C" | "D",
    "explanation": "one sentence why this answer is correct"
  }}
]
"""
    if router_instance is None:
        raw_response = llm_router.generate(prompt, force_local=True, api_key=api_key)
    else:
        raw_response = router_instance.generate(prompt)

    # Detect router-level errors
    if not raw_response or not raw_response.strip() or raw_response.startswith("ERROR:"):
        raise RuntimeError(raw_response or "LLM returned an empty response. Check that Ollama is running.")

    # Clean markdown fences
    text = raw_response.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1 :]
    if "```" in text:
        text = text[: text.rfind("```")]
    text = text.strip()

    try:
        questions = json.loads(text)
    except Exception as e:
        print(f"[Quiz Engine] Failed to parse questions JSON: {e}")
        # Safe fallback MCQs
        questions = [
            {
                "question": "Which of the following is a key feature of Python's clean code architecture?",
                "skill": focus_skills[0] if focus_skills else "Python",
                "difficulty": "medium",
                "type": "conceptual",
                "options": {
                    "A": "Readability and indentation enforcement",
                    "B": "Forced compile-time typing",
                    "C": "Hardware level memory address registers",
                    "D": "Absence of runtime dynamic libraries",
                },
                "correct": "A",
                "explanation": "Python prioritizes readability and enforces scope clean lines through semantic indentation.",
            }
        ]

    # Validate — ensure correct answer exists in options
    validated = []
    for q in questions:
        if (
            q.get("correct") in ["A", "B", "C", "D"]
            and q.get("options")
            and q["correct"] in q["options"]
        ):
            # Map key format for app compatibility if needed
            q["skill_tested"] = q.get("skill", "General")
            validated.append(q)

    return validated


def score_quiz(questions: list, user_answers: dict) -> dict:
    """
    Score completed quiz. user_answers = {question_index: "A"/"B"/"C"/"D"}
    Returns detailed results with per-skill breakdown.
    """
    total = len(questions)
    correct = 0
    wrong = []
    skill_scores = {}

    for i, q in enumerate(questions):
        skill = q.get("skill_tested", q.get("skill", "General"))
        if skill not in skill_scores:
            skill_scores[skill] = {"correct": 0, "total": 0}
        skill_scores[skill]["total"] += 1

        user_ans = user_answers.get(i)
        if user_ans == q["correct"]:
            correct += 1
            skill_scores[skill]["correct"] += 1
        else:
            wrong.append(
                {
                    "question": q["question"],
                    "your_answer": f"{user_ans}: {q['options'].get(user_ans, 'No answer')}",
                    "correct_answer": f"{q['correct']}: {q['options'][q['correct']]}",
                    "explanation": q.get("explanation", ""),
                    "skill": skill,
                    "type": q.get("type", ""),
                }
            )

    # Per-skill percentage
    skill_results = {}
    for skill, data in skill_scores.items():
        pct = round(data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        skill_results[skill] = {
            "score": pct,
            "correct": data["correct"],
            "total": data["total"],
            "grade": "Strong" if pct >= 80 else "Needs work" if pct >= 50 else "Weak area",
        }

    overall_pct = round(correct / total * 100) if total > 0 else 0

    return {
        "total_questions": total,
        "correct": correct,
        "wrong": wrong,
        "overall_pct": overall_pct,
        "overall_grade": (
            "A"
            if overall_pct >= 85
            else "B"
            if overall_pct >= 70
            else "C"
            if overall_pct >= 55
            else "D"
        ),
        "skill_results": skill_results,
        "weak_skills": [s for s, r in skill_results.items() if r["score"] < 60],
        "strong_skills": [s for s, r in skill_results.items() if r["score"] >= 80],
    }


# Curated resource clips per skill topic
SKILL_RESOURCES = {
    "Python": [
        {
            "title": "Python Internals You Must Know",
            "url": "https://www.youtube.com/watch?v=C25j1Q0r8Jg",
            "duration": "18 min",
        },
        {
            "title": "Advanced Python Decorators & Generators",
            "url": "https://www.youtube.com/watch?v=cKPlPJyQrt4",
            "duration": "32 min",
        },
    ],
    "FastAPI": [
        {
            "title": "FastAPI Full Course — Async, Dependency Injection & DBs",
            "url": "https://www.youtube.com/watch?v=tLKKmouUams",
            "duration": "45 min",
        }
    ],
    "Machine Learning": [
        {
            "title": "Production ML Engineering Best Practices",
            "url": "https://www.youtube.com/watch?v=pDMC25ZP1c8",
            "duration": "28 min",
        }
    ],
    "YOLO": [
        {
            "title": "YOLO Object Detection Pipeline & Architecture",
            "url": "https://www.youtube.com/watch?v=tFNJGim3FXw",
            "duration": "22 min",
        }
    ],
    "Docker": [
        {
            "title": "Docker Networking, Volumes & Cache Layers Explained",
            "url": "https://www.youtube.com/watch?v=pTFZFxd4hOI",
            "duration": "20 min",
        }
    ],
}


def get_resources_for_weak_skills(weak_skills: list) -> list:
    resources = []
    for skill in weak_skills:
        # Fuzzy match
        for key in SKILL_RESOURCES:
            if key.lower() in skill.lower() or skill.lower() in key.lower():
                for r in SKILL_RESOURCES[key]:
                    resources.append({"skill": skill, **r})
    return resources


def generate_quiz(resume_text: str, num_questions: int = 10,
                  api_key: str = None, use_ollama: bool = False) -> dict:
    """
    Returns:
      {
        "questions":      list of MCQ dicts,
        "needs_ollama":   bool,   # True if user asked >10 but Ollama wasn't used
        "total_requested": int,
        "total_generated": int,
        "error":          str | None
      }
    """
    needs_ollama   = num_questions > GEMINI_CAP and not use_ollama and not api_key
    actual_generate = num_questions if (use_ollama or api_key) else min(num_questions, GEMINI_CAP)
    force_local     = use_ollama and num_questions > GEMINI_CAP

    prompt = f"""You are a senior technical interviewer. Generate exactly {actual_generate} \
multiple choice questions based ONLY on skills and projects in this resume.

Resume:
{resume_text[:3500]}

RULES:
- Each question must test practical usage of a tool actually listed in the resume
- Do NOT ask definitions — ask scenario/debugging/architecture questions
- Do NOT repeat the same technology twice across questions
- 4 options per question labeled A, B, C, D with exactly one correct answer
- Difficulty: 60% intermediate, 40% advanced
- Cover variety: different tools, different projects

Return ONLY valid JSON. No markdown fences. No preamble. No explanation.

{{
  "questions": [
    {{
      "id": 1,
      "skill_tested": "FastAPI",
      "question": "<scenario question>",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct": "B",
      "explanation": "<why B is correct, referencing the resume context>"
    }}
  ]
}}"""

    try:
        raw = llm_router.generate(prompt, force_local=force_local, api_key=api_key)
    except Exception as e:
        return {
            "questions": [], "needs_ollama": needs_ollama,
            "total_requested": num_questions, "total_generated": 0,
            "error": f"Generation failed: {e}"
        }

    # Clean markdown fences if model adds them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw[raw.find("\n") + 1:]
    if raw.endswith("```"):
        raw = raw[:raw.rfind("```")]

    # Parse JSON
    data = None
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                pass

    if not data:
        return {
            "questions": [], "needs_ollama": needs_ollama,
            "total_requested": num_questions, "total_generated": 0,
            "error": "Could not parse quiz response. Try again."
        }

    questions = data.get("questions", [])
    # Align key names just in case
    for q in questions:
        if "skill_tested" not in q:
            q["skill_tested"] = q.get("skill", "General")

    return {
        "questions":       questions,
        "needs_ollama":    needs_ollama,
        "total_requested": num_questions,
        "total_generated": len(questions),
        "error":           None
    }
