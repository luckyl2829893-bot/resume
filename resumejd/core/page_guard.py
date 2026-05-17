"""
core/page_guard.py

CRITICAL: Prevents resume from expanding beyond original page count.
The #1 failure mode of AI resume tailors is turning a 1-page resume into 2 pages.

Strategy:
1. Measure original resume page count and character density.
2. Set a hard character budget for the tailored version.
3. After tailoring, if output exceeds budget -> trigger compression pass.
4. Compression: shorten bullet points, remove redundant phrases, tighten language
   WITHOUT removing any achievement, skill, or keyword.
"""

import re
import sys
from pathlib import Path

# Setup path mappings for config/routing compatibility
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router

# Empirical constants (A4 page, 10pt Helvetica, 0.75in margins)
CHARS_PER_LINE = 95  # approx characters per line
LINES_PER_PAGE = 52  # approx lines per A4 page (with normal spacing)
CHARS_PER_PAGE = CHARS_PER_LINE * LINES_PER_PAGE  # ~4940 chars

SAFETY_MARGIN = 0.92  # use 92% of page to avoid edge overflow


def estimate_page_count(text: str) -> float:
    """Estimate how many pages a resume text will occupy when rendered."""
    if not text:
        return 0.0
    lines = 0
    for line in text.split("\n"):
        if not line.strip():
            lines += 0.4  # blank line = partial line
        else:
            lines += max(1, len(line) / CHARS_PER_LINE)
    return lines / LINES_PER_PAGE


def get_char_budget(original_text: str) -> tuple[int, float]:
    """Returns the maximum character count the tailored resume is allowed to have."""
    original_pages = estimate_page_count(original_text)
    # Round to nearest 0.5 — a 1.1 page resume stays 1 page
    target_pages = max(1, round(original_pages * 2) / 2)
    budget = int(target_pages * CHARS_PER_PAGE * SAFETY_MARGIN)
    return budget, target_pages


COMPRESSION_SYSTEM = """You are a resume editor specializing in concise, high-impact writing.
Your job is to SHORTEN the resume to fit within a strict character limit
WITHOUT removing any skill, achievement, tool, keyword, or job entry.

Rules:
- Shorten bullet points: remove filler words, combine redundant points.
- Never delete an entire bullet — shorten it instead.
- Never remove a project, job entry, or skill section.
- Use strong action verbs to say more in fewer words.
- Maintain all quantified metrics (numbers, percentages, $ amounts).
Return ONLY the shortened resume text. No commentary."""


def compress_to_budget(text: str, budget: int, router_instance=None) -> str:
    """If tailored text exceeds budget, run a compression pass."""
    current_chars = len(text)
    if current_chars <= budget:
        return text  # Already fits — no compression needed

    overage_pct = round((current_chars - budget) / budget * 100, 1)

    prompt = f"""
The resume below is {current_chars} characters but must fit in {budget} characters.
That is {overage_pct}% over the limit — you need to shorten it by approximately {current_chars - budget} characters.

DO NOT remove any job, project, skill, or achievement.
DO shorten bullet points, remove filler phrases, tighten language.
Every keyword and metric must be preserved.

Resume to compress:
{text}
"""
    # Use global router fallback if router_instance is None
    if router_instance is None:
        raw_response = llm_router.generate(prompt)
    else:
        raw_response = router_instance.generate(prompt)

    # Clean markdown fences
    compressed = raw_response.strip()
    if compressed.startswith("```"):
        compressed = compressed[compressed.find("\n") + 1 :]
    if "```" in compressed:
        compressed = compressed[: compressed.rfind("```")]
    compressed = compressed.strip()

    # Safety check — if still over budget, do a second pass
    if len(compressed) > budget * 1.05:  # allow 5% tolerance
        prompt2 = f"""
Still {len(compressed) - budget} chars over budget. Remove filler, shorten further.
Keep ALL jobs, projects, skills, metrics.
{compressed}
"""
        if router_instance is None:
            raw_response2 = llm_router.generate(prompt2)
        else:
            raw_response2 = router_instance.generate(prompt2)

        compressed2 = raw_response2.strip()
        if compressed2.startswith("```"):
            compressed2 = compressed2[compressed2.find("\n") + 1 :]
        if "```" in compressed2:
            compressed2 = compressed2[: compressed2.rfind("```")]
        compressed = compressed2.strip()

    return compressed


def enforce_page_limit(original_text: str, tailored_text: str, router_instance=None) -> dict:
    """
    Main entry point. Called after tailoring.
    Returns the final text guaranteed to fit original page count.
    """
    budget, target_pages = get_char_budget(original_text)
    original_pages = estimate_page_count(original_text)
    tailored_pages = estimate_page_count(tailored_text)

    was_compressed = False
    final_text = tailored_text

    if len(tailored_text) > budget:
        final_text = compress_to_budget(tailored_text, budget, router_instance)
        was_compressed = True

    final_pages = estimate_page_count(final_text)

    return {
        "final_text": final_text,
        "original_pages": round(original_pages, 2),
        "tailored_pages_before": round(tailored_pages, 2),
        "final_pages": round(final_pages, 2),
        "target_pages": target_pages,
        "was_compressed": was_compressed,
        "chars_original": len(original_text),
        "chars_final": len(final_text),
        "budget": budget,
    }
