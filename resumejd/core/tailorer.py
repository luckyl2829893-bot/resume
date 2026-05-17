import sys
from pathlib import Path

# Enforce standard package search paths
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router
from core.ats_scorer import score

SYSTEM_TAILORER = """You are a world-class resume writer and ATS optimization expert.
You write resumes that consistently pass ATS filters AND impress human recruiters.
You follow the Google X-Y-Z formula: "Accomplished [X] as measured by [Y] by doing [Z]."
You NEVER fabricate experience. You ONLY enhance, reorder, and reframe what is already there.
You maintain the candidate's authentic voice while maximizing relevance to the target role."""

TAILOR_PROMPT = """
TASK: Rewrite the candidate's resume to be perfectly tailored for the target job.

RULES:
1. NEVER invent experience, skills, or achievements that don't exist in the original.
2. CRITICAL: Output sections in EXACTLY THIS ORDER: {section_order_instruction}
   Do NOT reorder sections. Keep them exactly as listed above.
3. DO rewrite bullet points using the Google X-Y-Z formula with quantified impact ("Accomplished [X], measured by [Y], by doing [Z]") where possible.
4. DO naturally weave in ALL critical keywords and required skills from the JD (listed below) — but never keyword-stuff.
5. DO adjust the summary/objective to directly address what this employer needs.
6. Return the rewritten resume in this exact structured format:
   NAME: <full name>
   CONTACT: <phone> | <email> | LinkedIn | GitHub
   ---SECTION: <SECTION NAME>---
   <content>
   Use • for bullets. For each project use ##PROJECT: <project name> on its own line.
   Do not include any other conversational preamble or markdown code blocks.

TARGET JOB DETAILS:
Title: {job_title}
Company: {company}
Domain: {domain}
Seniority: {seniority}

CRITICAL KEYWORDS TO WEAVE IN:
{critical_keywords}

REQUIRED SKILLS TO HIGHLIGHT:
{required_skills}

KEY RESPONSIBILITIES:
{responsibilities}

ORIGINAL RESUME TEXT:
{resume_text}

Now write the tailored resume. Start directly with "NAME:".
"""


def format_tailored_output(name: str, contact: str, sections_dict: dict,
                            section_order: list = None) -> str:
    """
    Assembles the final structured resume text.
    If section_order is provided, sections are output in that order (original resume order).
    Any sections not in section_order are appended at the end.
    """
    lines = []
    lines.append(f"NAME: {name.strip()}")
    lines.append(f"CONTACT: {contact.strip()}")

    rendered = set()

    if section_order:
        for canonical in section_order:
            canonical_up = canonical.upper()
            # Find best match in sections_dict
            matched_key = None
            for key in sections_dict:
                key_up = key.upper()
                if canonical_up == key_up:
                    matched_key = key
                    break
            if not matched_key:
                for key in sections_dict:
                    key_up = key.upper()
                    if canonical_up in key_up or key_up in canonical_up:
                        matched_key = key
                        break
            if matched_key and matched_key not in rendered:
                content = sections_dict[matched_key].strip()
                if content:
                    lines.append(f"---SECTION: {canonical_up}---")
                    lines.append(content)
                    rendered.add(matched_key)

    # Append any remaining sections not in section_order
    for key, content in sections_dict.items():
        if key not in rendered and content.strip():
            lines.append(f"---SECTION: {key.upper().strip()}---")
            lines.append(content.strip())

    return "\n".join(lines)


def tailor(resume_text: str, jd_dict: dict, jd_raw: str,
           layout_profile: dict = None,
           force_local: bool = False,
           force_gemini: bool = False) -> str:
    """
    Step 1: Tailor the resume using the Google X-Y-Z formula.
    Step 2: Score the tailored version.
    Step 3: Refinement pass if score < 75.
    Step 4: Reorder sections to match original resume layout (using layout_profile).

    Returns: Structured tailored resume text in NAME/CONTACT/SECTION format.
    """
    from core.resume_builder import parse_tailored_sections

    # Determine section order to enforce
    section_order = None
    section_order_instruction = "PROFILE SUMMARY, PROFESSIONAL SKILL, EDUCATION, PROJECTS, INTERESTS"
    if layout_profile and layout_profile.get("section_order"):
        section_order = layout_profile["section_order"]
        section_order_instruction = ", ".join(section_order)

    critical_kws = ", ".join(jd_dict.get("keywords_critical", []))
    req_skills = ", ".join(jd_dict.get("required_skills", []))
    responsibilities = "\n".join(f"- {r}" for r in jd_dict.get("responsibilities", [])[:8])

    prompt = TAILOR_PROMPT.format(
        section_order_instruction=section_order_instruction,
        job_title=jd_dict.get("job_title", "Software Developer"),
        company=jd_dict.get("company", "Acme Corp"),
        domain=jd_dict.get("domain", "General Tech"),
        seniority=jd_dict.get("seniority", "mid"),
        critical_keywords=critical_kws if critical_kws else "None specified",
        required_skills=req_skills if req_skills else "None specified",
        responsibilities=responsibilities if responsibilities else "General software development tasks",
        resume_text=resume_text,
    )

    tailored_text = llm_router.generate(prompt, force_local=force_local, force_gemini=force_gemini)

    # 2. Re-score
    scores = score(tailored_text, jd_dict)

    # 3. Refinement pass if score < 75
    if scores["total_score"] < 75 and (scores["missing_keywords"] or scores["missing_skills"]):
        refinement_prompt = f"""
You are refining a resume that scores {scores['total_score']}/100 on ATS matching.
Incorporate the following missing keywords and skills organically — do NOT fabricate experience.
Missing Keywords: {', '.join(scores['missing_keywords'][:10])}
Missing Skills: {', '.join(scores['missing_skills'][:5])}

CRITICAL: Output sections in EXACTLY this order: {section_order_instruction}

Return the optimized resume in the exact same structured format:
NAME: <full name>
CONTACT: <phone> | <email> | LinkedIn | GitHub
---SECTION: <SECTION NAME>---
<content>
Use • for bullets. For projects use ##PROJECT: <name> on its own line.
No preamble. No markdown fences.

Current tailored resume:
{tailored_text}
"""
        tailored_text = llm_router.generate(refinement_prompt, force_local=force_local, force_gemini=force_gemini)

    # 4. Post-process: parse and reorder sections to match layout_profile
    parsed = parse_tailored_sections(tailored_text)

    # Recover name from original if LLM left it blank
    if not parsed["name"] or parsed["name"] == "Candidate Name":
        orig_lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
        if orig_lines:
            parsed["name"] = orig_lines[0].replace("#", "").replace("**", "").strip()

    # Recover contact from original if LLM dropped it
    if not parsed["contact"]:
        for l in resume_text.split("\n"):
            l_strip = l.strip()
            if any(term in l_strip.lower() for term in ["|", "linkedin", "github", "email", "@"]):
                parsed["contact"] = l_strip.replace("**", "").replace("*", "").strip()
                break

    return format_tailored_output(
        parsed["name"],
        parsed["contact"],
        parsed["sections"],
        section_order=section_order,
    )
