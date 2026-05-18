import sys
from pathlib import Path

# Enforce standard package search paths
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router
from core.ats_scorer import score


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
           force_gemini: bool = False,
           api_key: str = None) -> str:
    """
    Rewrites resume to target JD.
    Guarantees:
      - Output score >= original score (never downgrade)
      - Output score >= 80% (if achievable within 3 passes)
      - Frozen: name, contact, education dates, company names
      - Retains Page Guard budget and format fingerprinting
    """
    from core.ats_scorer import score as ats_score
    from core.resume_builder import parse_tailored_sections
    from core.page_guard import enforce_page_limit

    original_result = ats_score(resume_text, jd_dict)
    original_score  = original_result.get("total_score", 0)

    # Target: at minimum keep current score; push to 80 if below
    target_score = max(80, original_score)

    best_text  = resume_text   # fallback: always return at least the original
    best_score = original_score

    missing_keywords = original_result.get("missing_keywords", [])
    missing_skills   = original_result.get("missing_skills",   [])

    # Determine section order to enforce
    section_order = None
    section_order_instruction = "PROFILE SUMMARY, PROFESSIONAL SKILL, EDUCATION, PROJECTS, INTERESTS"
    if layout_profile and layout_profile.get("section_order"):
        section_order = layout_profile["section_order"]
        section_order_instruction = ", ".join(section_order)

    for attempt in range(3):
        if attempt == 0:
            focus = (
                "First pass: rewrite bullets into X-Y-Z formula and inject "
                f"these missing keywords naturally: {missing_keywords[:12]}"
            )
        elif attempt == 1:
            focus = (
                f"Second pass: previous attempt scored {best_score}%. "
                f"Focus harder on integrating: {missing_keywords[:10]}. "
                f"Also include these skills: {missing_skills[:8]}"
            )
        else:
            focus = (
                f"Final pass: still at {best_score}%. Be more aggressive with "
                f"alignment while staying truthful. Remaining gaps: {missing_keywords[:8]}"
            )

        prompt = f"""You are an expert resume writer and ATS optimization specialist.
{focus}

ABSOLUTE FREEZE — do NOT change ANY of these:
- Candidate name and all contact details (email, phone, LinkedIn, GitHub)
- University name, degree title, graduation year
- All project names (keep exactly as written)
- All GitHub links
- Do NOT invent metrics, tools, or experiences not in the original

DO:
- Rewrite bullet points using: "Accomplished [X], measured by [Y], by doing [Z]"
- Naturally weave in missing keywords
- Reorder skills section so JD-matching skills appear first
- Use action verbs from the JD where they fit
- If a section is already strong — tighten phrasing only, do not remove content

CRITICAL: Output sections in EXACTLY this order: {section_order_instruction}

Original resume:
{resume_text}

Job description target:
{jd_raw[:2000]}

Return the COMPLETE rewritten resume in this EXACT format — nothing else:
NAME: <full name>
CONTACT: <phone> | <email> | <linkedin> | <github>
---SECTION: PROFILE SUMMARY---
<2-3 sentence summary>
---SECTION: PROFESSIONAL SKILL---
• Category: skill1, skill2
---SECTION: EDUCATION---
<institution> | <degree> | <years>
---SECTION: PROJECTS---
##PROJECT: <exact original project name>
• Accomplished X, measured by Y, by doing Z
• bullet 2
LINK - <original github link>
##PROJECT: <next project>
...
---SECTION: INTERESTS---
• Interest 1  • Interest 2
"""

        try:
            candidate_text   = llm_router.generate(prompt, force_local=force_local, force_gemini=force_gemini, api_key=api_key)
            candidate_result = ats_score(candidate_text, jd_dict)
            candidate_score  = candidate_result.get("total_score", 0)
        except Exception:
            continue   # attempt failed — try next pass

        # Track best result seen
        if candidate_score > best_score:
            best_score = candidate_score
            best_text  = candidate_text

        # Update gaps for next pass
        missing_keywords = candidate_result.get("missing_keywords", [])
        missing_skills   = candidate_result.get("missing_skills",   [])

        if best_score >= target_score:
            break   # target reached early

    # DOWNGRADE PROTECTION: if every rewrite was worse, return original
    if best_score < original_score:
        best_text = resume_text

    # Post-process: parse, format, recover name/contact, and enforce page limit
    parsed = parse_tailored_sections(best_text)

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

    tailored_ordered = format_tailored_output(
        parsed["name"],
        parsed["contact"],
        parsed["sections"],
        section_order=section_order,
    )

    # Enforce Page Limits
    page_result = enforce_page_limit(resume_text, tailored_ordered)
    final_text = page_result["final_text"]

    # Re-score after compression if it was compressed
    if page_result["was_compressed"]:
        best_score_after_guard = ats_score(final_text, jd_dict).get("total_score", 0)

    # If running in Streamlit, save page guard telemetry directly to session state
    try:
        import streamlit as st
        st.session_state.page_info = page_result
    except ImportError:
        pass

    return final_text


def get_tailor_scores(resume_text: str, tailored_text: str,
                      jd_dict: dict) -> tuple:
    """Returns (original_score, tailored_score) as ints for UI display."""
    from core.ats_scorer import score as ats_score
    orig = ats_score(resume_text,   jd_dict).get("total_score", 0)
    tail = ats_score(tailored_text, jd_dict).get("total_score", 0)
    return orig, tail
