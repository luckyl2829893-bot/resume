"""
resumejd Autonomous Job Application Agent v2
=============================================
Runs fully locally via Ollama. No external API calls.
Dry Run Mode is active by default — will NOT submit until user opts in.

Steps:
  1. Discover jobs from LinkedIn, Naukri, Internshala
  2. ATS-bypass resume generation per job
  3. AI-screener-bypass cover letter generation
  4. Headless Playwright form filling & submission
  5. Anti-bot evasion + rate limiting + full audit log
"""

import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import asyncio
import json
import random
import re
import sys
import os
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
sys.path.append(str(Path(__file__).parent.parent))

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from fake_useragent import UserAgent

from core.llm_router import generate
from core.jd_parser import parse_jd
from core.ats_scorer import score
from core.tailorer import tailor
from core.resume_builder import build_pdf
from db.tracker import add_application

# ─── Directory Setup ──────────────────────────────────────────────────────────
AGENT_DIR = Path("agent")
SCREENSHOTS_DIR = AGENT_DIR / "screenshots"
RESUMES_DIR = AGENT_DIR / "resumes"
LOGS_DIR = AGENT_DIR / "logs"
for _d in [SCREENSHOTS_DIR, RESUMES_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_PER_RUN = 15
DELAYS = (45, 180)  # seconds between applications (human-pacing range)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 Safari/605.1.15",
]


# ─── Logging ──────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─── STEP 1: Job Discovery ─────────────────────────────────────────────────────

async def discover_linkedin(page, job_title: str, location: str) -> list:
    """Scrapes LinkedIn public job search (Easy Apply filter only — no login needed)."""
    jobs = []
    try:
        url = (
            f"https://www.linkedin.com/jobs/search/"
            f"?keywords={job_title.replace(' ', '%20')}"
            f"&location={location.replace(' ', '%20')}"
            f"&f_AL=true"  # Easy Apply filter
        )
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(3000)
        cards = await page.query_selector_all(".job-search-card")
        for card in cards[:20]:
            try:
                title_el = await card.query_selector(".base-search-card__title")
                company_el = await card.query_selector(".base-search-card__subtitle")
                link_el = await card.query_selector("a.base-card__full-link")
                if title_el and company_el and link_el:
                    jobs.append({
                        "platform": "linkedin",
                        "role": (await title_el.inner_text()).strip(),
                        "company": (await company_el.inner_text()).strip(),
                        "url": await link_el.get_attribute("href"),
                        "apply_url": await link_el.get_attribute("href"),
                        "jd_text": "",
                        "status": "discovered",
                    })
            except Exception:
                continue
    except Exception as e:
        log(f"LinkedIn discovery error: {e}")
    return jobs


async def discover_naukri(page, job_title: str, location: str) -> list:
    """Scrapes Naukri.com job listings (Indian market)."""
    jobs = []
    try:
        slug_title = job_title.lower().replace(" ", "-")
        slug_loc = location.lower().replace(" ", "-")
        url = f"https://www.naukri.com/{slug_title}-jobs-in-{slug_loc}"
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(3000)
        # Naukri uses various card classes across versions — try both
        cards = await page.query_selector_all(".jobTuple, .job-tuple")
        for card in cards[:15]:
            try:
                title_el = await card.query_selector(".title, .job-title")
                company_el = await card.query_selector(".companyInfo span, .company-name")
                link_el = await card.query_selector("a.title, a.job-title")
                if title_el and company_el and link_el:
                    jobs.append({
                        "platform": "naukri",
                        "role": (await title_el.inner_text()).strip(),
                        "company": (await company_el.inner_text()).strip(),
                        "url": await link_el.get_attribute("href"),
                        "apply_url": await link_el.get_attribute("href"),
                        "jd_text": "",
                        "status": "discovered",
                    })
            except Exception:
                continue
    except Exception as e:
        log(f"Naukri discovery error: {e}")
    return jobs


async def discover_internshala(page, job_title: str) -> list:
    """Scrapes Internshala job listings (fresher / intern roles)."""
    jobs = []
    try:
        slug_title = job_title.lower().replace(" ", "-")
        url = f"https://internshala.com/jobs/{slug_title}-jobs/"
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(3000)
        cards = await page.query_selector_all(".internship_meta, .job-internship-card")
        for card in cards[:10]:
            try:
                title_el = await card.query_selector(".profile, .job-title-text")
                company_el = await card.query_selector(".company_name, .company-name")
                link_el = await card.query_selector("a")
                if title_el and company_el and link_el:
                    href = await link_el.get_attribute("href") or ""
                    full_url = (
                        f"https://internshala.com{href}"
                        if href.startswith("/") else href
                    )
                    jobs.append({
                        "platform": "internshala",
                        "role": (await title_el.inner_text()).strip(),
                        "company": (await company_el.inner_text()).strip(),
                        "url": full_url,
                        "apply_url": full_url,
                        "jd_text": "",
                        "status": "discovered",
                    })
            except Exception:
                continue
    except Exception as e:
        log(f"Internshala discovery error: {e}")
    return jobs


async def scrape_jd_text(page, url: str) -> str:
    """Navigates to a job URL and extracts the visible body text (up to 4000 chars)."""
    try:
        await page.goto(url, timeout=20000)
        await page.wait_for_timeout(2000)
        body = await page.query_selector("body")
        return (await body.inner_text())[:4000] if body else ""
    except Exception:
        return ""


# ─── STEP 2: ATS Bypass Resume Generation ─────────────────────────────────────

def build_ats_bypassed_resume(
    resume_text: str,
    jd_text: str,
    layout_profile: dict = None,
) -> bytes:
    """
    Generates a fully tailored, ATS-optimised PDF for a specific job.
    Applies keyword density injection, verb alignment, and skills reordering.
    """
    jd_dict = parse_jd(jd_text)
    score_result = score(resume_text, jd_dict)
    tailored = tailor(resume_text, jd_dict, jd_text)

    # Additional ATS bypass pass via Ollama
    missing_kw = score_result.get("missing_keywords", [])
    action_verbs = jd_dict.get("action_verbs", [])
    bypass_prompt = f"""
You are an ATS optimization expert. Improve the tailored resume below by applying:
1. Every keyword in this list must appear AT LEAST 2 times naturally: {missing_kw}
2. Reorder the Skills section so the most JD-matching skills appear first.
3. Replace action verbs in bullets to match these JD verbs where applicable: {action_verbs}
4. Ensure ALL dates are in MM/YYYY format.
5. Remove any special characters except: • - | and standard punctuation.

Resume to optimize:
{tailored}

Return ONLY the optimized resume in the same NAME:/CONTACT:/---SECTION: format.
No explanation. No preamble.
"""
    bypassed = generate(bypass_prompt, force_local=True)
    return build_pdf(bypassed, "tailored_resume.pdf", layout_profile)


# ─── STEP 3: AI Screener Bypass Cover Letter ──────────────────────────────────

def build_bypass_cover_letter(
    resume_text: str,
    jd_text: str,
    company: str,
    role: str,
) -> str:
    """
    Generates a human-sounding cover letter designed to evade AI screening tools.
    Uses perplexity variance injection and specificity anchoring.
    """
    prompt = f"""
You are a professional writer creating a cover letter that will pass AI screening tools.

Rules you MUST follow:
1. VARY sentence length deliberately: mix 6-word punchy sentences with 25-35 word complex
   compound sentences — never uniform rhythm throughout the letter.
2. Open with something SPECIFIC to this company or role — NOT "I am writing to express..."
3. Inject 2-3 hyper-specific details from the JD: exact team names, product names,
   or specific tools mentioned.
4. Include natural first-person phrases: "I noticed in your JD that...",
   "Having worked directly with X...", "One thing that stood out to me..."
5. Include ONE deliberate minor grammatical informality (e.g. starting a sentence with "And").
6. Do NOT use bullet points anywhere in the letter.
7. Exactly 3 paragraphs: Hook + 2 quantified achievements + Forward close.
8. 200-250 words total.

Context:
Company: {company}
Role: {role}
JD: {jd_text[:1500]}
Resume: {resume_text[:1500]}

Write only the letter body. No subject line. No "Dear Hiring Manager".
Start directly with the opening sentence.
"""
    return generate(prompt, force_local=True)


# ─── STEP 4: Form Filling & Submission ────────────────────────────────────────

async def fill_linkedin_easy_apply(
    page,
    candidate: dict,
    resume_path: str,
    cover_letter: str,
    dry_run: bool = True,
) -> bool:
    """Fills and optionally submits a LinkedIn Easy Apply form."""
    try:
        await page.goto(candidate["apply_url"], timeout=30000)
        await page.wait_for_timeout(2000)

        # Click Easy Apply button
        easy_btn = (
            await page.query_selector('[aria-label*="Easy Apply"]')
            or await page.query_selector(".jobs-apply-button")
        )
        if easy_btn:
            await easy_btn.click()
            await page.wait_for_timeout(2000)

        # Phone number
        phone_field = await page.query_selector('input[id*="phoneNumber"]')
        if phone_field:
            await human_type(phone_field, candidate.get("phone", ""))

        # Resume upload
        file_input = await page.query_selector('input[type="file"]')
        if file_input:
            await file_input.set_input_files(resume_path)
            await page.wait_for_timeout(1500)

        # Cover letter textarea
        cl_area = (
            await page.query_selector('textarea[id*="cover"]')
            or await page.query_selector('textarea[aria-label*="cover"]')
        )
        if cl_area:
            await human_type(cl_area, cover_letter)

        # Handle "Why do you want to work here?" style open-text fields
        why_fields = await page.query_selector_all("textarea")
        for field in why_fields:
            placeholder = await field.get_attribute("placeholder") or ""
            label = await field.get_attribute("aria-label") or ""
            combined = (placeholder + label).lower()
            if any(kw in combined for kw in ["why", "tell us", "motivat", "interest"]):
                answer = generate(
                    f"Answer in 150 words: {label or placeholder}\n"
                    f"Context: applying for {candidate['role']} at {candidate['company']}."
                    f"\nResume snippet: {candidate.get('resume_snippet', '')}",
                    force_local=True,
                )
                await human_type(field, answer)

        # Screenshot before final action
        safe_name = (
            f"{candidate['company']}_{candidate['role']}"
            .replace(" ", "_")
            .replace("/", "_")[:80]
        )
        screenshot_path = str(SCREENSHOTS_DIR / f"{safe_name}.png")
        await page.screenshot(path=screenshot_path)
        log(f"Screenshot saved: {screenshot_path}")

        if dry_run:
            log(f"DRY RUN — LinkedIn: {candidate['role']} @ {candidate['company']}")
            return True

        # Submit
        submit_btn = (
            await page.query_selector('[aria-label="Submit application"]')
            or await page.query_selector('button[type="submit"]')
        )
        if submit_btn:
            await submit_btn.click()
            await page.wait_for_timeout(2000)
            log(f"SUBMITTED: {candidate['role']} @ {candidate['company']}")
            return True

    except PWTimeout:
        log(f"Timeout on {candidate.get('company', '?')} — skipping")
    except Exception as e:
        log(f"LinkedIn form error for {candidate.get('company', '?')}: {e}")
    return False


async def fill_naukri_apply(
    page,
    candidate: dict,
    resume_path: str,
    cover_letter: str,
    dry_run: bool = True,
) -> bool:
    """Fills and optionally submits a Naukri apply form."""
    try:
        await page.goto(candidate["apply_url"], timeout=30000)
        await page.wait_for_timeout(2500)

        apply_btn = (
            await page.query_selector("#apply-button")
            or await page.query_selector(".apply-button")
        )
        if apply_btn:
            await apply_btn.click()
            await page.wait_for_timeout(2000)

        file_input = await page.query_selector('input[type="file"]')
        if file_input:
            await file_input.set_input_files(resume_path)
            await page.wait_for_timeout(1000)

        safe_name = (
            f"naukri_{candidate['company']}_{candidate['role']}"
            .replace(" ", "_")[:80]
        )
        screenshot_path = str(SCREENSHOTS_DIR / f"{safe_name}.png")
        await page.screenshot(path=screenshot_path)

        if dry_run:
            log(f"DRY RUN — Naukri: {candidate['role']} @ {candidate['company']}")
            return True

        submit = await page.query_selector('[type="submit"]')
        if submit:
            await submit.click()
            await page.wait_for_timeout(1500)
            log(f"SUBMITTED Naukri: {candidate['role']} @ {candidate['company']}")
            return True

    except Exception as e:
        log(f"Naukri apply error for {candidate.get('company', '?')}: {e}")
    return False


async def fill_internshala_apply(
    page,
    candidate: dict,
    resume_path: str,
    cover_letter: str,
    dry_run: bool = True,
) -> bool:
    """Fills and optionally submits an Internshala apply form."""
    try:
        await page.goto(candidate["apply_url"], timeout=30000)
        await page.wait_for_timeout(2500)

        apply_btn = await page.query_selector(".apply_now_button, #apply-button")
        if apply_btn:
            await apply_btn.click()
            await page.wait_for_timeout(2000)

        # Cover letter on Internshala
        cl_area = await page.query_selector("textarea#cover_letter_text, textarea.cover-letter")
        if cl_area:
            await human_type(cl_area, cover_letter[:500])  # Internshala has char limits

        safe_name = (
            f"internshala_{candidate['company']}_{candidate['role']}"
            .replace(" ", "_")[:80]
        )
        screenshot_path = str(SCREENSHOTS_DIR / f"{safe_name}.png")
        await page.screenshot(path=screenshot_path)

        if dry_run:
            log(f"DRY RUN — Internshala: {candidate['role']} @ {candidate['company']}")
            return True

        submit = await page.query_selector('[type="submit"], .submit_button')
        if submit:
            await submit.click()
            await page.wait_for_timeout(1500)
            log(f"SUBMITTED Internshala: {candidate['role']} @ {candidate['company']}")
            return True

    except Exception as e:
        log(f"Internshala apply error for {candidate.get('company', '?')}: {e}")
    return False


# ─── STEP 5: Orchestration ────────────────────────────────────────────────────

async def create_stealth_context(pw):
    """
    Launches Chromium with anti-bot fingerprint spoofing.
    Hides webdriver flag, spoofs plugins, sets realistic locale/timezone.
    """
    browser = await pw.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-infobars",
            "--window-size=1366,768",
        ]
    )
    ctx = await browser.new_context(
        user_agent=random.choice(USER_AGENTS),
        viewport={"width": 1366, "height": 768},
        locale="en-IN",
        timezone_id="Asia/Kolkata",
        extra_http_headers={
            "Accept-Language":  "en-IN,en;q=0.9,hi;q=0.8",
            "Accept-Encoding":  "gzip, deflate, br",
            "Accept":           "text/html,application/xhtml+xml,*/*;q=0.8",
            "Connection":       "keep-alive",
        }
    )
    await ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins',   { get: () => [1,2,3,4,5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-IN','en','hi'] });
        window.chrome = { runtime:{}, loadTimes:function(){}, csi:function(){}, app:{} };
        const _pq = window.navigator.permissions.query.bind(navigator.permissions);
        window.navigator.permissions.query = p =>
            p.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : _pq(p);
    """)
    return browser, ctx


async def human_type(element, text: str):
    """Types text with random per-character delays (50-150ms) to mimic human input."""
    await element.click()
    await asyncio.sleep(random.uniform(0.3, 0.7))
    for char in text:
        await element.type(char, delay=random.randint(50, 150))
        if random.random() < 0.05:            # occasional hesitation
            await asyncio.sleep(random.uniform(0.3, 0.9))


async def human_scroll(page, direction: str = "down", steps: int = None):
    """Scrolls page in human-like increments before interacting."""
    steps = steps or random.randint(2, 5)
    for _ in range(steps):
        amt = random.randint(200, 500) * (1 if direction == "down" else -1)
        await page.evaluate(f"window.scrollBy(0, {amt})")
        await asyncio.sleep(random.uniform(0.4, 1.1))


async def run_agent(
    config: dict,
    resume_text: str,
    layout_profile: dict = None,
    progress_callback=None,
) -> dict:
    """
    Main agent entry point.

    config keys:
      job_title, location, experience_level,
      max_apps, platforms, dry_run, candidate_info
    """
    job_title = config["job_title"]
    location = config.get("location", "India")
    dry_run = config.get("dry_run", True)  # Safety default: never submit without opt-in
    max_apps = min(config.get("max_apps", 10), MAX_PER_RUN)
    platforms = config.get("platforms", ["linkedin", "naukri", "internshala"])
    candidate = config.get("candidate_info", {})

    all_jobs = []
    applied_count = 0

    log(
        f"Agent starting — job: '{job_title}' | location: '{location}' | "
        f"dry_run: {dry_run} | max_apps: {max_apps} | platforms: {platforms}"
    )

    async with async_playwright() as pw:
        browser, ctx = await create_stealth_context(pw)
        page = await ctx.new_page()

        # ── Phase 1: Discovery ───────────────────────────────────────────────
        log("Phase 1: Discovering jobs...")

        if "linkedin" in platforms:
            try:
                jobs = await discover_linkedin(page, job_title, location)
                all_jobs.extend(jobs)
                log(f"LinkedIn: found {len(jobs)} jobs")
            except Exception as e:
                log(f"LinkedIn discovery failed entirely: {e}")

        if "naukri" in platforms:
            try:
                jobs = await discover_naukri(page, job_title, location)
                all_jobs.extend(jobs)
                log(f"Naukri: found {len(jobs)} jobs")
            except Exception as e:
                log(f"Naukri discovery failed entirely: {e}")

        if "internshala" in platforms:
            try:
                jobs = await discover_internshala(page, job_title)
                all_jobs.extend(jobs)
                log(f"Internshala: found {len(jobs)} jobs")
            except Exception as e:
                log(f"Internshala discovery failed entirely: {e}")

        log(f"Total discovered: {len(all_jobs)} jobs")

        # Save all discovered jobs to the tracker
        for job in all_jobs:
            try:
                add_application(
                    job["company"], job["role"], job.get("url", ""),
                    status="discovered",
                    notes=f"Discovered via {job['platform']} agent.",
                )
            except Exception:
                pass

        if progress_callback:
            progress_callback("discovered", len(all_jobs))

        # ── Phase 2: Tailor + Apply ──────────────────────────────────────────
        log("Phase 2: Tailoring resumes and applying...")

        for i, job in enumerate(all_jobs[:max_apps]):
            if applied_count >= max_apps:
                break

            log(
                f"Processing {i+1}/{min(len(all_jobs), max_apps)}: "
                f"{job['role']} @ {job['company']} [{job['platform']}]"
            )

            # Scrape full JD text
            job["jd_text"] = await scrape_jd_text(page, job["url"])
            if not job["jd_text"] or len(job["jd_text"]) < 100:
                log(f"Could not scrape JD for {job['company']} — skipping")
                continue

            # Build ATS-bypassed tailored resume PDF
            try:
                resume_pdf = build_ats_bypassed_resume(
                    resume_text, job["jd_text"], layout_profile
                )
                safe_name = (
                    f"{job['company']}_{job['role']}"
                    .replace(" ", "_")
                    .replace("/", "_")[:80]
                )
                resume_path = str(RESUMES_DIR / f"{safe_name}.pdf")
                with open(resume_path, "wb") as f:
                    f.write(resume_pdf)
                log(f"Tailored resume saved: {resume_path}")
            except Exception as e:
                log(f"Resume build failed for {job['company']}: {e}")
                continue

            # Generate AI-screener-bypass cover letter
            cover = build_bypass_cover_letter(
                resume_text, job["jd_text"], job["company"], job["role"]
            )
            candidate_with_job = {
                **candidate,
                **job,
                "resume_snippet": resume_text[:300],
            }

            # Apply via platform-specific handler
            success = False
            try:
                if job["platform"] == "linkedin":
                    success = await fill_linkedin_easy_apply(
                        page, candidate_with_job, resume_path, cover, dry_run
                    )
                elif job["platform"] == "naukri":
                    success = await fill_naukri_apply(
                        page, candidate_with_job, resume_path, cover, dry_run
                    )
                elif job["platform"] == "internshala":
                    success = await fill_internshala_apply(
                        page, candidate_with_job, resume_path, cover, dry_run
                    )
                else:
                    log(
                        f"Platform '{job['platform']}' form-fill not yet implemented — "
                        f"resume saved at {resume_path}"
                    )
                    success = True  # Count discovery+tailor as a success
            except Exception as e:
                log(f"Apply orchestration error for {job.get('company', '?')}: {e}")

            if success:
                applied_count += 1
                status = "applied" if not dry_run else "dry_run"
                try:
                    add_application(
                        job["company"], job["role"], job.get("url", ""),
                        status=status,
                        notes=f"Agent run. Resume: {resume_path}. Dry run: {dry_run}.",
                    )
                except Exception:
                    pass

                if progress_callback:
                    progress_callback("applied", applied_count)

            # Human-like pacing between applications
            if i < min(len(all_jobs), max_apps) - 1:
                delay = random.randint(*DELAYS)
                log(f"Waiting {delay}s before next application (human pacing)...")
                await asyncio.sleep(delay)

        await browser.close()

    log(
        f"Agent complete. Total discovered: {len(all_jobs)} | "
        f"{'Applied' if not dry_run else 'Processed (dry run)'}: {applied_count}"
    )
    return {
        "discovered": len(all_jobs),
        "applied": applied_count,
        "dry_run": dry_run,
    }
