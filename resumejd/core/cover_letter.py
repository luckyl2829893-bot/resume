import sys
from pathlib import Path

# Setup standard path mappings for config/routing compatibility
sys.path.append(str(Path(__file__).parent.parent))
from core import llm_router

SYSTEM_CL = """You are an expert cover letter writer and executive recruiter.
You write highly engaging, professional, and personalized 3-paragraph cover letters.
Never be generic. Never start with "I am writing to apply..." or "To whom it may concern".
Keep it under 350 words, human, professional, and impactful."""

COVER_LETTER_PROMPT = """
Write a compelling 3-paragraph cover letter for a candidate applying to:
Company: {company_name}
Job Description:
{jd_raw}

CANDIDATE'S RESUME (extract accomplishments and metrics from here):
{resume_text}

STRUCTURE REQUIREMENTS:
- Paragraph 1: An engaging hook that outlines interest in the specific role, referencing {company_name} and aligning with their mission/culture. Do NOT use boilerplate openings.
- Paragraph 2: Highlight 2-3 specific, quantified achievements directly from the candidate's resume that solve or address key needs in the Job Description. Include percentages, dollars, or metrics.
- Paragraph 3: A confident, forward-looking close expressing excitement about contributing to the team, and a polite call to action for an interview.

Output ONLY the cover letter text, with no extra placeholders, explanations, or metadata.
"""

def generate(resume_text: str, jd_raw: str, company_name: str, api_key: str = None) -> str:
    """
    Generates a high-precision, 3-paragraph tailored cover letter using llm_router.
    """
    prompt = COVER_LETTER_PROMPT.format(
        company_name=company_name,
        jd_raw=jd_raw,
        resume_text=resume_text
    )
    return llm_router.generate(prompt, api_key=api_key)
