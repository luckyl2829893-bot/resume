import warnings
# Silence verbose third-party warnings from transformers and python-docx
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# setup relative path additions
sys.path.append(str(Path(__file__).parent))

import requests
from core.llm_router import generate, check_ollama_status, get_last_model_used
from core.resume_parser import parse_resume, extract_layout_profile
from core.jd_parser import parse_jd, scrape_jd
from core.ats_scorer import score
from core.tailorer import tailor
from core.cover_letter import generate as generate_cl
from core.interview_prep import generate_prep, generate_deep_project_drill
from core.resume_builder import build_pdf, build_docx
import db.tracker as tracker
from db.tracker import log_experiment, get_experiments

# Initialize tracking database on boot
tracker.init_db()

# --- Page Settings & Global Themes ---
st.set_page_config(
    page_title="Resume AI — ATS Optimizer Workspace",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global custom styles for unified card alignment
st.markdown("""
<style>
    .metric-card {
        background-color: #121820;
        border: 1px solid #00ffcc22;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.9em;
        color: #e2e8f088;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# Load environmental configs
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- PERSISTENT STATE INITIALIZATION ---
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "tailored_resume" not in st.session_state:
    st.session_state.tailored_resume = ""
if "parsed_jd" not in st.session_state:
    st.session_state.parsed_jd = {}
if "raw_jd" not in st.session_state:
    st.session_state.raw_jd = ""
if "original_score_dict" not in st.session_state:
    st.session_state.original_score_dict = {}
if "tailored_score_dict" not in st.session_state:
    st.session_state.tailored_score_dict = {}
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "role_title" not in st.session_state:
    st.session_state.role_title = ""
if "force_local_global" not in st.session_state:
    st.session_state.force_local_global = False
if "layout_profile" not in st.session_state:
    st.session_state.layout_profile = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Auto (Gemini → Ollama fallback)"

# --- SIDEBAR CONTROL PANEL ---
# ── Ollama status check — cached 60s ─────────────────────────────────────────
@st.cache_data(ttl=60)
def check_ollama_fast() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

ollama_ok = check_ollama_fast()

with st.sidebar:
    st.title("📄 Resume AI")
    st.caption("Privacy-First Local ATS Rewrite Workspace")
    st.divider()

    # ── 4-screen navigation ────────────────────────────────────────────────
    page = st.radio("Navigate Workspace", [
        "🏠 Resume Tailorer",
        "📨 Cover Letter",
        "🎤 Interview Prep",
        "📊 Job Tracker",
    ])
    st.divider()

    # ── Model selector ────────────────────────────────────────────────
    st.session_state.model_choice = st.radio(
        "LLM Engine",
        [
            "Auto (Gemini → Ollama fallback)",
            "Force Gemini only",
            "Force Ollama only",
        ],
        index=["Auto (Gemini → Ollama fallback)", "Force Gemini only", "Force Ollama only"].index(
            st.session_state.model_choice
        ),
    )
    st.divider()

    # ── Live active-model indicator ───────────────────────────────────────
    last_model = get_last_model_used()
    lm = last_model['model']
    ll = last_model['label']
    if lm == 'gemini':
        st.markdown(f"**Active Engine:** {ll}")
    elif lm in ('ollama', 'ollama_fallback'):
        st.markdown(f"**Active Engine:** {ll}")
    elif 'error' in lm or 'failed' in lm:
        st.markdown(f"⚠️ **Active Engine:** {ll}")
    else:
        st.markdown("**Active Engine:** Not run yet")

    gemini_key = os.getenv("GEMINI_API_KEY")
    st.markdown(f"{'🟢' if ollama_ok else '🔴'} **Ollama:** {'Online' if ollama_ok else 'Offline'}")
    st.markdown(f"{'🟢' if gemini_key else '🔴'} **Gemini Key:** {'Set' if gemini_key else 'Missing'}")
    st.markdown("🟢 **MiniLM:** Loaded (CPU)")
    st.divider()

    # ── ⚙️ System Config & MLOps Sidebar Expander ─────────────────────────
    with st.expander("⚙️ Settings & MLOps"):
        tab_cfg, tab_mlops = st.tabs(["Config", "MLOps Log"])
        with tab_cfg:
            st.caption("API Key (.env)")
            gemini_key_input = st.text_input(
                "Gemini API Key",
                value=os.getenv("GEMINI_API_KEY", ""),
                type="password",
                label_visibility="collapsed",
            )
            if st.button("Save Key"):
                try:
                    env_file = Path(__file__).parent / ".env"
                    with open(env_file, "w") as f:
                        f.write(f"GEMINI_API_KEY={gemini_key_input}\n")
                    st.success("Saved! Restart app.")
                except Exception as e:
                    st.error(f"Failed: {e}")

            st.caption("Ollama Status")
            if st.button("Ping Ollama"):
                health = check_ollama_status()
                if health["running"]:
                    st.success("Ollama Online!")
                    st.caption(f"Installed: {', '.join(health['models'])}")
                else:
                    st.error("Ollama Offline")

        with tab_mlops:
            experiments = get_experiments(limit=15)
            if not experiments:
                st.caption("No runs logged yet.")
            else:
                for exp in experiments:
                    delta = exp.get("score_delta", 0)
                    with st.expander(f"{exp.get('company','?')[:10]} (+{delta:.0f})"):
                        st.write(f"**Score:** {exp.get('tailored_score',0):.1f}")
                        st.write(f"**Model:** {exp.get('model_used','?')}")
                        if exp.get("missing_keywords"):
                            st.caption(f"Gaps: {', '.join(exp['missing_keywords'][:3])}")

    st.divider()
    st.caption("resumejd v2 — Local AI Resume Optimizer")

# ── SCREEN 1: RESUME TAILORER (MAIN ENGINE) ──
if page == "🏠 Resume Tailorer":
    st.title("🏠 Resume Tailorer & Matcher")
    st.caption("Upload your master resume, paste or scrape a job description, and watch the system tailor bullet metrics in seconds.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Candidate Resume")
        resume_file = st.file_uploader("Select resume file (PDF or DOCX)", type=["pdf", "docx"], help="Your primary master resume file")
        
        if resume_file:
            try:
                file_bytes = resume_file.read()
                resume_file.seek(0)
                st.session_state.resume_text = parse_resume(resume_file)
                # Extract layout fingerprint for format mirroring
                if resume_file.name.lower().endswith(".pdf"):
                    st.session_state.layout_profile = extract_layout_profile(file_bytes)
                    accent_hex = st.session_state.layout_profile.get("accent_color_hex", "N/A")
                    st.success(f"✓ Resume loaded ({len(st.session_state.resume_text)} chars) | Accent: #{accent_hex}")
                else:
                    st.success(f"✓ Resume loaded ({len(st.session_state.resume_text)} chars)")
                with st.expander("Preview extracted resume text"):
                    st.text(st.session_state.resume_text[:1000] + "...")
            except Exception as e:
                st.error(f"Text extraction failed: {e}")
        
    with col2:
        st.subheader("2. Paste or Scrape Job Details")
        jd_input_choice = st.radio("Posting Input Type", ["Paste raw text", "Auto-Scrape Posting URL"], horizontal=True)
        
        if jd_input_choice == "Paste raw text":
            jd_text_input = st.text_area("Paste job posting description here", height=180)
            jd_url_input = st.text_input("Posting URL (optional)", "")
        else:
            jd_url_input = st.text_input("Enter Job Posting URL (Indeed, LinkedIn, etc.)")
            jd_text_input = ""
            if jd_url_input:
                if st.button("Auto-Fetch Job details"):
                    with st.spinner("Scraping job posting..."):
                        try:
                            scraped_content = scrape_jd(jd_url_input)
                            st.session_state.raw_jd = scraped_content
                            st.success("✓ Job description extracted successfully!")
                        except Exception as e:
                            st.error(f"URL Scraping failed: {e}")
                            
        # Keep text synchronized
        if jd_input_choice == "Paste raw text" and jd_text_input:
            st.session_state.raw_jd = jd_text_input
            
        if st.session_state.raw_jd:
            with st.expander("Preview extracted Job description"):
                st.text(st.session_state.raw_jd[:1000] + "...")
                
    st.divider()
    
    # Process reframe trigger
    if st.button("🚀 Tailor and Match Resume", type="primary", use_container_width=True):
        if not st.session_state.resume_text:
            st.error("Please upload your master resume file first.")
        elif not st.session_state.raw_jd:
            st.error("Please provide a target Job Description first.")
        else:
            with st.spinner("Parsing job description into structured requirements..."):
                try:
                    st.session_state.parsed_jd = parse_jd(st.session_state.raw_jd)
                    st.session_state.company_name = st.session_state.parsed_jd.get("company") or "Target Company"
                    st.session_state.role_title = st.session_state.parsed_jd.get("job_title") or "Software Engineer"
                except Exception as e:
                    st.error(f"Failed to structure JD: {e}")
                    
            with st.spinner("Computing initial ATS match scores..."):
                st.session_state.original_score_dict = score(st.session_state.resume_text, st.session_state.parsed_jd)
                
            with st.spinner("Rewriting bullet metrics using Google X-Y-Z formula... (takes 30-45s)"):
                try:
                    _mc = st.session_state.model_choice
                    _fl = (_mc == "Force Ollama only")
                    _fg = (_mc == "Force Gemini only")

                    st.session_state.tailored_resume = tailor(
                        st.session_state.resume_text,
                        st.session_state.parsed_jd,
                        st.session_state.raw_jd,
                        layout_profile=st.session_state.layout_profile,
                        force_local=_fl,
                        force_gemini=_fg,
                    )
                    st.session_state.tailored_score_dict = score(
                        st.session_state.tailored_resume, st.session_state.parsed_jd
                    )

                    # ── MLOps: log this experiment run ──
                    from core.llm_router import get_last_model_used as _glmu
                    _lm = _glmu()
                    _ts = st.session_state.tailored_score_dict
                    _lp = st.session_state.layout_profile or {}
                    try:
                        log_experiment(
                            model_used=_lm["model"],
                            model_label=_lm["label"],
                            job_title=st.session_state.role_title,
                            company=st.session_state.company_name,
                            original_score=st.session_state.original_score_dict.get("total_score", 0),
                            tailored_score=_ts.get("total_score", 0),
                            grade=_ts.get("grade", "?"),
                            missing_keywords=_ts.get("missing_keywords", []),
                            section_order=_lp.get("section_order", []),
                            run_type="tailor",
                        )
                    except Exception:
                        pass  # Never let MLOps logging crash the main workflow

                    st.success(f"Resume tailored using **{_lm['label']}**!")
                except Exception as e:
                    st.error(f"Tailoring failed: {e}")

                    
    # Render improvement analysis if ready
    if st.session_state.tailored_resume:
        st.subheader("📊 ATS Score Improvement Analysis")
        
        # metric card rows
        m_col1, m_col2, m_col3 = st.columns(3)
        orig_s = st.session_state.original_score_dict.get("total_score", 0.0)
        tail_s = st.session_state.tailored_score_dict.get("total_score", 0.0)
        
        m_col1.metric("Original Match Score", f"{orig_s}/100")
        m_col2.metric("Optimized Match Score", f"{tail_s}/100", delta=f"+{tail_s - orig_s:.1f}")
        m_col3.metric("ATS Match Tier Grade", st.session_state.tailored_score_dict.get("grade", "D"))
        
        st.subheader("Match Criteria Breakdowns")
        break_col1, break_col2 = st.columns(2)
        with break_col1:
            st.caption("Original Score Metrics Breakdown")
            for key, val in st.session_state.original_score_dict.get("breakdown", {}).items():
                st.progress(val / 100, f"{key.title()}: {val}%")
        with break_col2:
            st.caption("Tailored Score Metrics Breakdown")
            for key, val in st.session_state.tailored_score_dict.get("breakdown", {}).items():
                st.progress(val / 100, f"{key.title()}: {val}%")
                
        # Gap notifications
        miss_k = st.session_state.tailored_score_dict.get("missing_keywords", [])
        miss_s = st.session_state.tailored_score_dict.get("missing_skills", [])
        if miss_k or miss_s:
            st.warning(f"⚠️ **Residual ATS Gaps:** The following parameters could not be safely tailored without inventing experience:\n"
                       f"* **Missing Keywords:** {', '.join(miss_k[:6]) if miss_k else 'None'}\n"
                       f"* **Missing Skills:** {', '.join(miss_s[:6]) if miss_s else 'None'}")
                       
        # Display side-by-side review + edits
        st.divider()
        st.subheader("📝 Review & Tweak Tailored Resume")
        st.caption("Review the Google X-Y-Z formatted bullet points below. Edit the output manually in the text panel if necessary.")
        
        edited_text = st.text_area("Resume Content Output", st.session_state.tailored_resume, height=500)
        if edited_text != st.session_state.tailored_resume:
            st.session_state.tailored_resume = edited_text
            
        # Download files row
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            try:
                pdf_bytes = build_pdf(
                    st.session_state.tailored_resume,
                    layout_profile=st.session_state.layout_profile
                )
                st.download_button(
                    "⬇️ Download ATS PDF",
                    data=pdf_bytes,
                    file_name=f"Resume_{st.session_state.company_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF build crashed: {e}")
        with dl_col2:
            try:
                docx_bytes = build_docx(
                    st.session_state.tailored_resume,
                    layout_profile=st.session_state.layout_profile
                )
                st.download_button(
                    "⬇️ Download ATS DOCX",
                    data=docx_bytes,
                    file_name=f"Resume_{st.session_state.company_name.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"DOCX build crashed: {e}")
        with dl_col3:
            if st.button("💾 Save Application to Tracker", use_container_width=True):
                tracker.add_application(
                    company=st.session_state.company_name,
                    role=st.session_state.role_title,
                    url=jd_url_input if jd_input_choice == "Auto-Scrape Posting URL" else "",
                    status="draft",
                    original_score=orig_s,
                    tailored_score=tail_s,
                    notes="Optimized via Resume AI."
                )
                st.success("✓ Application saved to Local SQLite Job Tracker!")

# ── SCREEN 2: COVER LETTER ──
elif page == "\U0001f4e8 Cover Letter":
    st.title("\U0001f4e8 Cover Letter Generator")
    st.caption("Generate a high-converting, 3-paragraph tailored cover letter that matches your optimized resume highlights to the role's culture.")

    if not st.session_state.tailored_resume or not st.session_state.raw_jd:
        st.info("Please optimize your resume on the Tailorer screen first to pre-populate inputs automatically.")
    else:
        st.subheader("Tailor Cover Letter")
        c_name = st.text_input("Confirm Company Name", st.session_state.company_name)

        if st.button("Generate Cover Letter", type="primary", use_container_width=True):
            with st.spinner("Writing cover letter..."):
                try:
                    _mc = st.session_state.model_choice
                    cl_result = generate_cl(
                        st.session_state.tailored_resume,
                        st.session_state.raw_jd,
                        c_name,
                    )
                    st.session_state.cover_letter_text = cl_result
                    lm = get_last_model_used()
                    st.success(f"Cover letter generated using **{lm['label']}**!")
                except Exception as e:
                    st.error(f"Cover letter generation failed: {e}")

        if "cover_letter_text" in st.session_state and st.session_state.cover_letter_text:
            st.subheader("📝 Cover Letter Output")
            edited_cl = st.text_area("Edit Cover Letter", st.session_state.cover_letter_text, height=350)
            if edited_cl != st.session_state.cover_letter_text:
                st.session_state.cover_letter_text = edited_cl

            st.download_button(
                "⬇️ Download Cover Letter (.txt)",
                data=st.session_state.cover_letter_text,
                file_name=f"Cover_Letter_{c_name.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ── SCREEN 3: INTERVIEW PREP ──
elif page == "\U0001f3a4 Interview Prep":
    st.title("\U0001f3a4 Custom Interview Preparation Kit")
    st.caption("Generate custom Behavioral questions, Technical questions, Smart follow-ups, and Red-flags tailored exactly to your resume context.")


    if not st.session_state.tailored_resume or not st.session_state.parsed_jd:
        st.info("⚠️ Please optimize your resume on the Tailorer screen first to compile structural prep questions.")
    else:
        # Decide tab layout based on Ollama availability
        if ollama_ok:
            tab_std, tab_drill = st.tabs(["Standard Prep", "Project Deep Drill 🔬"])
            st.sidebar.caption("🟢 Unlimited depth mode active (Ollama)")
        else:
            tab_std = st.container()
            tab_drill = None
            st.info("Start Ollama to unlock Deep Project Drill mode.", icon="🔬")

        with tab_std:
            if st.button("Generate Interview Preparation Kit", type="primary", use_container_width=True):
                with st.spinner("Analyzing role context and compiling prep kit..."):
                    try:
                        prep_kit = generate_prep(st.session_state.tailored_resume, st.session_state.parsed_jd)
                        st.session_state.prep_kit = prep_kit
                        st.success("✓ Prep Kit generated!")
                    except Exception as e:
                        st.error(f"Failed to generate prep kit: {e}")

            if "prep_kit" in st.session_state and st.session_state.prep_kit:
                p_kit = st.session_state.prep_kit
                tab1, tab2, tab3, tab4 = st.tabs(["💡 Behavioral (STAR)", "💻 Technical", "❓ Questions to Ask", "⚠️ Red Flags"])

                with tab1:
                    st.subheader("STAR Behavioral Questions")
                    for i, q in enumerate(p_kit.get("behavioral", [])):
                        with st.expander(f"Question {i+1}: {q.get('question')}"):
                            st.write(f"**Recruiter's Goal:** {q.get('why_asked')}")
                            st.info(f"**Suggested STAR Response:**\n{q.get('model_answer')}")
                with tab2:
                    st.subheader("Required Skills Technical Questions")
                    for i, q in enumerate(p_kit.get("technical", [])):
                        with st.expander(f"Q{i+1} ({q.get('topic')}): {q.get('question')}"):
                            st.write(q.get("expected_answer"))
                with tab3:
                    st.subheader("Smart Questions to Ask Interviewer")
                    for i, q in enumerate(p_kit.get("company", [])):
                        st.write(f"**{i+1}.** {q}")
                with tab4:
                    st.subheader("Interview Red Flags & Prep Focuses")
                    for r in p_kit.get("red_flags", []):
                        st.error(r)

        # ── Deep Project Drill tab (Ollama only) ──────────────────────────────
        if tab_drill is not None:
            with tab_drill:
                st.caption("Generates hyper-specific Q&A so you can explain every line of your code to a senior engineer.")

                drill_resume = st.session_state.get("resume_text", "") or st.session_state.tailored_resume
                if not drill_resume:
                    st.warning("Upload your resume on the Tailorer screen first.")
                else:
                    import re as _re
                    project_matches = _re.findall(r'##PROJECT:\s*(.+)', drill_resume)
                    if not project_matches:
                        # Fallback — look for bold-style project headers in the text
                        project_matches = _re.findall(
                            r'\n([A-Z][A-Za-z0-9 :&\-]{10,60})\n', drill_resume
                        )[:5] or ["My Main Project"]

                    selected_project = st.selectbox("Select Project to Drill", project_matches, key="drill_project")
                    quiz_mode = st.toggle("Quiz Me Mode (hide answers until revealed)", value=False, key="quiz_mode")

                    if st.button("Generate Deep Drill", type="primary", key="gen_drill"):
                        with st.spinner(f"Ollama generating drill for '{selected_project}'... (1-3 min)"):
                            drill = generate_deep_project_drill(drill_resume, selected_project, "ollama")
                        st.session_state.drill_kit = drill

                    if "drill_kit" in st.session_state and st.session_state.drill_kit:
                        drill = st.session_state.drill_kit
                        if "error" in drill:
                            st.error(drill["error"])
                            if "raw" in drill:
                                with st.expander("Raw LLM output"):
                                    st.code(drill["raw"])
                        else:
                            section_labels = {
                                "architecture": "🏗 Architecture Deep Dive (WHY questions)",
                                "internals": "🔧 Technology Internals (HOW it works)",
                                "failure_modes": "💥 Failure Modes & Tradeoffs",
                                "rapid_fire": "⚡ Rapid Fire — Prove You Built It",
                                "concept_gaps": "🧠 Concept Gap Fillers",
                            }
                            for key, label in section_labels.items():
                                items = drill.get(key, [])
                                if not items:
                                    continue
                                st.subheader(label)
                                for idx, item in enumerate(items):
                                    q_text = item.get("q", "")
                                    with st.expander(f"Q{idx+1}: {q_text[:90]}{'...' if len(q_text)>90 else ''}"):
                                        st.markdown(f"**Question:** {q_text}")
                                        if "term" in item:
                                            st.caption(f"Term detected in resume: `{item['term']}`")
                                        if quiz_mode:
                                            st.text_area("Your answer:", key=f"{key}_{idx}_ans", height=80)
                                            if st.button("Reveal Answer ▼", key=f"{key}_{idx}_rev"):
                                                st.markdown(f"**Model Answer:** {item.get('a', '')}")
                                                if "follow_up" in item:
                                                    st.info(f"**Senior follow-up:** {item['follow_up']}")
                                        else:
                                            st.markdown(f"**Answer:** {item.get('a', '')}")
                                            if "follow_up" in item:
                                                st.info(f"**Senior follow-up:** {item['follow_up']}")

# ── SCREEN 3: JOB TRACKER ──
elif page == "📊 Job Tracker":
    st.title("📊 SQLite Job Application Tracker")
    st.caption("Manage your applications and run the autonomous AI job application agent.")

    stats = tracker.get_stats()
    c_t, c_a, c_i, c_o = st.columns(4)
    c_t.metric("Total Saved", stats["total"])
    c_a.metric("Applied", stats["applied"])
    c_i.metric("Interviews", stats["interviews"])
    c_o.metric("Offers Received", stats["offers"])

    tab_board, tab_agent = st.tabs(["Application Board", "Auto-Apply Agent 🤖"])

    # ── Application Board tab ──────────────────────────────────────────────────
    with tab_board:
        apps = tracker.get_all()
        if not apps:
            st.info("No applications saved yet. Tailor a resume and save it from the Tailorer screen!")
        else:
            st.subheader("Your Application Pipeline")
            for app in apps:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <span style="font-size:1.3em;font-weight:bold;color:#00ffcc;">{app['company']}</span> —
                        <span style="font-size:1.1em;color:#fafafa;">{app['role']}</span>
                        <br/><span style="font-size:0.85em;color:#e2e8f066;">Status: {app['status']} | Updated: {app['updated_date']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Update / Delete"):
                        sc, nc = st.columns([1, 2])
                        status_list = ["draft", "discovered", "dry_run", "applied",
                                       "screening", "interview", "offer", "rejected", "accepted"]
                        try:
                            active_idx = status_list.index(app["status"])
                        except ValueError:
                            active_idx = 0
                        new_status = sc.selectbox("Status", status_list, index=active_idx, key=f"s_{app['id']}")
                        new_notes = nc.text_input("Notes", app["notes"] or "", key=f"n_{app['id']}")
                        a1, a2 = st.columns(2)
                        if a1.button("Save", key=f"sv_{app['id']}"):
                            tracker.update_status(app["id"], new_status, new_notes)
                            st.success("Saved!")
                            st.rerun()
                        if a2.button("🗑 Delete", key=f"dl_{app['id']}", type="secondary"):
                            tracker.delete(app["id"])
                            st.rerun()
                    st.divider()

    # ── Auto-Apply Agent tab ───────────────────────────────────────────────────
    with tab_agent:
        if not ollama_ok:
            st.error("Auto-Apply Agent requires Ollama to be running locally. Start Ollama and refresh.")
            st.stop()

        st.info(
            "Agent runs fully locally via Ollama. No data is sent externally.\n\n"
            "**Dry Run is ON by default** — it will discover jobs, tailor resumes, and take "
            "screenshots, but will NOT submit applications until you uncheck the safety toggle.",
            icon="ℹ️",
        )
        st.caption("First-time setup: run `playwright install chromium` in your terminal once.")

        col1, col2 = st.columns(2)
        with col1:
            job_title_agent = st.text_input("Target Job Title", placeholder="e.g. ML Engineer, Data Analyst")
            location_agent = st.text_input("Location", placeholder="e.g. Bangalore, India")
            max_apps = st.number_input("Max Applications per Run", min_value=1, max_value=15, value=5)
        with col2:
            st.selectbox("Experience Level", ["Fresher / Entry Level", "1-2 Years", "2-5 Years"])
            platforms = st.multiselect(
                "Platforms to Search",
                ["linkedin", "naukri", "internshala"],
                default=["linkedin", "naukri", "internshala"],
            )

        st.subheader("Candidate Info")
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            cand_name = st.text_input("Full Name", placeholder="Lakshya Sharma")
            cand_email = st.text_input("Email")
        with ci2:
            cand_phone = st.text_input("Phone")
            cand_linkedin = st.text_input("LinkedIn URL")
        with ci3:
            cand_github = st.text_input("GitHub URL")

        st.subheader("Safety Controls")
        sc1, sc2 = st.columns(2)
        with sc1:
            dry_run = st.checkbox("Dry Run Mode (discover + tailor, NO submit)", value=True)
            if not dry_run:
                st.warning("⚠️ Live mode ON — agent will submit real applications.")
        with sc2:
            min_delay = st.number_input("Min delay between apps (sec)", value=60, min_value=10)
            max_delay = st.number_input("Max delay between apps (sec)", value=180, min_value=30)

        agent_resume_file = st.file_uploader(
            "Upload Master Resume for Agent (PDF)",
            type=["pdf"],
            key="agent_resume",
        )

        stats_container = st.empty()
        log_container = st.empty()

        can_start = bool(job_title_agent and agent_resume_file)
        if st.button("🚀 Start Agent", type="primary", disabled=not can_start):
            from datetime import datetime as _dt
            import asyncio as _asyncio

            resume_bytes_agent = agent_resume_file.read()
            agent_resume_file.seek(0)
            agent_resume_text = parse_resume(agent_resume_file)
            agent_layout = extract_layout_profile(resume_bytes_agent)

            agent_config = {
                "job_title": job_title_agent,
                "location": location_agent or "India",
                "max_apps": int(max_apps),
                "platforms": platforms,
                "dry_run": dry_run,
                "candidate_info": {
                    "name": cand_name, "email": cand_email,
                    "phone": cand_phone, "linkedin": cand_linkedin,
                    "github": cand_github,
                },
            }

            _stats = {"discovered": 0, "applied": 0}

            def _progress(event_type, count):
                _stats[event_type] = count
                stats_container.markdown(
                    f"**Discovered:** {_stats['discovered']} &nbsp;|&nbsp; "
                    f"**{'Applied' if not dry_run else 'Processed'}:** {_stats['applied']}"
                )

            try:
                from agent.job_applicant_agent import run_agent
                with st.spinner("Agent running — do not close this tab..."):
                    result = _asyncio.run(run_agent(agent_config, agent_resume_text, agent_layout, _progress))

                st.success(
                    f"Agent complete! Discovered **{result['discovered']}** jobs. "
                    f"{'Applied' if not dry_run else 'Dry-run processed'}: **{result['applied']}**. "
                    f"Check the Application Board tab."
                )
            except ImportError:
                st.error("Playwright not installed. Run: `pip install playwright && playwright install chromium`")
            except Exception as _ae:
                st.error(f"Agent error: {_ae}")

            # Show today's log
            from pathlib import Path as _P
            _log_path = _P(f"agent/logs/{_dt.now().strftime('%Y-%m-%d')}.log")
            if _log_path.exists():
                log_container.text_area("Agent Log", _log_path.read_text(encoding="utf-8"), height=300)

