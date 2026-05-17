"""
core/template_engine.py

5 built-in resume templates with defined layouts and content capacity limits.
Each template has a maximum content capacity (in characters per section).
If the user's content exceeds a template's capacity -> warn and offer options.
"""

import json
import sys
from pathlib import Path

# Setup path mappings for config/routing compatibility
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router

TEMPLATES = {
    "modern_minimal": {
        "name": "Modern Minimal",
        "description": "Clean single-column, lots of white space. Best for: design, UX, creative roles.",
        "thumbnail_emoji": "🎨",
        "page_count": 1,
        "style": "single_column",
        "section_capacity": {
            "header": 200,  # Name + contact only
            "summary": 300,  # Short punchy summary
            "experience": 1200,  # ~3 jobs, 2 bullets each
            "skills": 400,  # Inline comma-separated
            "education": 250,
            "projects": 600,  # 2 projects max
        },
        "total_capacity": 2950,
        "suitable_for": ["design", "ux", "creative", "marketing", "junior"],
        "not_suitable_for": ["senior", "research", "academic", "many_jobs"],
    },
    "classic_two_col": {
        "name": "Classic Two-Column",
        "description": "Skills/education sidebar + experience main column. Best for: tech, engineering.",
        "thumbnail_emoji": "💼",
        "page_count": 1,
        "style": "two_column",
        "section_capacity": {
            "header": 180,
            "summary": 250,
            "experience": 1600,  # More space in main column
            "skills": 500,  # Sidebar can hold more categorized skills
            "education": 300,
            "projects": 500,
            "certifications": 200,
        },
        "total_capacity": 3530,
        "suitable_for": ["tech", "engineering", "data_science", "mid_level"],
        "not_suitable_for": ["creative", "academic"],
    },
    "compact_fresher": {
        "name": "Compact Fresher",
        "description": "Maximizes content density. Best for: freshers with lots of projects/internships.",
        "thumbnail_emoji": "🎓",
        "page_count": 1,
        "style": "compact",
        "section_capacity": {
            "header": 180,
            "summary": 200,
            "education": 400,  # Education first for freshers
            "projects": 1200,  # Projects are primary — lots of space
            "internships": 600,
            "skills": 600,  # Technical skills prominent
            "achievements": 300,
            "certifications": 200,
        },
        "total_capacity": 3680,
        "suitable_for": ["fresher", "student", "intern", "entry_level"],
        "not_suitable_for": ["senior", "manager", "10+_years"],
    },
    "senior_executive": {
        "name": "Senior Executive",
        "description": "2-page, leadership-focused, metrics-heavy. Best for: senior/lead/manager roles.",
        "thumbnail_emoji": "👔",
        "page_count": 2,
        "style": "executive",
        "section_capacity": {
            "header": 250,
            "summary": 600,  # Extended executive summary
            "experience": 3200,  # Many jobs, detailed bullets
            "leadership": 600,
            "skills": 400,
            "education": 300,
            "projects": 800,
            "publications": 400,
        },
        "total_capacity": 6550,
        "suitable_for": ["senior", "lead", "manager", "director", "8+_years"],
        "not_suitable_for": ["fresher", "student", "1-2_years"],
    },
    "tech_focused": {
        "name": "Tech Focused",
        "description": "GitHub-style, skills matrix, project showcase. Best for: SWE, ML, DevOps.",
        "thumbnail_emoji": "💻",
        "page_count": 1,
        "style": "tech",
        "section_capacity": {
            "header": 200,
            "skills_matrix": 800,  # Categorized: Languages | Frameworks | Tools | Cloud
            "experience": 1400,
            "projects": 900,  # GitHub links, tech stack per project
            "education": 250,
            "certifications": 200,
            "open_source": 300,
        },
        "total_capacity": 4050,
        "suitable_for": ["software_engineer", "ml_engineer", "devops", "backend", "fullstack"],
        "not_suitable_for": ["non_tech", "creative", "finance"],
    },
}


def check_content_fit(resume_text: str, template_id: str) -> dict:
    """
    Check if the user's resume content fits inside a template's capacity.
    Returns fit analysis and recommendations.
    """
    if template_id not in TEMPLATES:
        return {"fits": False, "message": "Unknown template"}

    template = TEMPLATES[template_id]
    total_capacity = template["total_capacity"]
    content_length = len(resume_text)

    overflow_chars = content_length - total_capacity
    overflow_pct = round(overflow_chars / total_capacity * 100, 1)

    fits = content_length <= total_capacity

    if fits:
        utilization = round(content_length / total_capacity * 100, 1)
        status = "perfect" if utilization >= 85 else "spacious"
        message = (
            f"Content fits perfectly ({utilization}% capacity used)."
            if utilization >= 85
            else f"Content fits with room to spare ({utilization}% used). "
            f"Consider adding more detail to projects or skills."
        )
    else:
        status = "overflow"
        message = (
            f"Content is {overflow_pct}% over this template's capacity "
            f"({overflow_chars} extra characters). "
            f"Options: switch to a larger template, or compress content."
        )

    # Suggest alternative templates if overflow
    alternatives = []
    if not fits:
        for tid, t in TEMPLATES.items():
            if tid != template_id and t["total_capacity"] >= content_length:
                alternatives.append(
                    {
                        "id": tid,
                        "name": t["name"],
                        "emoji": t["thumbnail_emoji"],
                        "capacity": t["total_capacity"],
                        "headroom": t["total_capacity"] - content_length,
                        "pages": t["page_count"],
                    }
                )
        alternatives.sort(key=lambda x: x["headroom"])  # closest fit first

    return {
        "template_id": template_id,
        "template_name": template["name"],
        "fits": fits,
        "status": status,
        "content_length": content_length,
        "capacity": total_capacity,
        "overflow_chars": max(0, overflow_chars),
        "overflow_pct": max(0, overflow_pct),
        "message": message,
        "alternative_templates": alternatives,
    }


def fill_template(resume_text: str, template_id: str, router_instance=None) -> dict:
    """
    Restructure the user's resume content to match a template's layout and sections.
    Does not change any core information — only reorganizes and reformats structure.
    """
    if template_id not in TEMPLATES:
        return {"success": False, "error": "invalid_template", "message": "Unknown template"}

    template = TEMPLATES[template_id]

    # First check if it fits
    fit = check_content_fit(resume_text, template_id)

    if not fit["fits"]:
        return {"success": False, "fit_analysis": fit, "error": "overflow", "message": fit["message"]}

    sections_str = "\n".join(
        f"- {sec}: max {cap} chars" for sec, cap in template["section_capacity"].items()
    )

    prompt = f"""
You are a professional resume formatter.
Restructure the resume below into the "{template['name']}" template format.

TEMPLATE LAYOUT: {template['style']}
TEMPLATE SECTIONS (with max character limits per section):
{sections_str}

RULES:
1. Do NOT change any core content — rearrange and reformat only.
2. Reorder sections to match the template's priority (listed above = priority order).
3. Keep ALL information — nothing deleted.
4. Use the section names exactly as listed above as headers.
5. Format skills for this template's style:
   - single_column: comma-separated inline.
   - two_column: categorized (Languages | Frameworks | Tools).
   - tech: full skills matrix with categories.
   - compact: grouped, space-efficient.
6. Return ONLY the formatted resume text.

Original resume:
{resume_text}
"""
    if router_instance is None:
        raw_response = llm_router.generate(prompt)
    else:
        raw_response = router_instance.generate(prompt)

    # Clean markdown fences
    formatted_text = raw_response.strip()
    if formatted_text.startswith("```"):
        formatted_text = formatted_text[formatted_text.find("\n") + 1 :]
    if "```" in formatted_text:
        formatted_text = formatted_text[: formatted_text.rfind("```")]
    formatted_text = formatted_text.strip()

    return {
        "success": True,
        "formatted_text": formatted_text,
        "template": template,
        "fit_analysis": fit,
        "model_used": "llm-router",
    }


def get_template_gallery() -> list:
    """Returns template list for display in UI."""
    return [
        {
            "id": tid,
            "name": t["name"],
            "emoji": t["thumbnail_emoji"],
            "description": t["description"],
            "pages": t["page_count"],
            "capacity": t["total_capacity"],
            "suitable_for": t["suitable_for"],
        }
        for tid, t in TEMPLATES.items()
    ]
