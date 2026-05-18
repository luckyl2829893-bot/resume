"""
core/skill_quiz.py

Resume-grounded skill quiz generator and test runner.

Anti-hallucination strategy:
- Extract skills/tools/projects ONLY from the resume text.
- Each question batch (20-30 Qs) is generated from a specific skill scope.
- Questions include the skill name + project/context from the resume as grounding.
- LLM is instructed to generate questions a person could only answer if they
  ACTUALLY worked with that technology — not surface-level definitions.
- All correct answers are stored and verified at generation time.
- After test: score by skill domain + recommend YouTube clips for weak areas.
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


def extract_resume_skills(resume_text: str, router_instance=None) -> dict:
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
        raw_response = llm_router.generate(prompt, force_local=True)
    else:
        raw_response = router_instance.generate(prompt)

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
    skills_profile: dict, focus_skills: list, num_questions: int, router_instance=None
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
        raw_response = llm_router.generate(prompt, force_local=True)
    else:
        raw_response = router_instance.generate(prompt)

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
        skill = q.get("skill", "General")
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
