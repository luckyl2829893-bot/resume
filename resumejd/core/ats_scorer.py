import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Pre-load sentence transformer layout similarity model (MiniLM-L6) on CPU
# Ensures fast computation without memory conflicts with Ollama/YOLO GPU schedulers
model = SentenceTransformer("all-MiniLM-L6-v2")

def score(resume_text: str, jd_dict: dict) -> dict:
    """
    Computes a weighted ATS compatibility score (0-100) comparing a candidate's resume
    against structured Job Description metadata.
    
    Weights Breakdown:
    * 35% — exact critical keyword match rate
    * 25% — required skills coverage
    * 20% — MiniLM semantic cosine similarity (all-MiniLM-L6-v2)
    * 10% — section presence check (experience, education, skills, projects)
    * 10% — metric/number density in bullets
    """
    resume_lower = resume_text.lower()
    
    # 1. Exact critical keywords match (35%)
    critical_keywords = [k.lower() for k in jd_dict.get("keywords_critical", []) if k.strip()]
    matched_critical = []
    missed_critical = []
    
    if critical_keywords:
        for kw in critical_keywords:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, resume_lower):
                matched_critical.append(kw)
            else:
                missed_critical.append(kw)
        keyword_match_rate = len(matched_critical) / len(critical_keywords)
    else:
        keyword_match_rate = 1.0 # fallback default
        
    keyword_score = keyword_match_rate * 100
    
    # 2. Required skills match (25%)
    required_skills = [s.lower() for s in jd_dict.get("required_skills", []) if s.strip()]
    matched_skills = []
    missed_skills = []
    
    if required_skills:
        for skill in required_skills:
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, resume_lower):
                matched_skills.append(skill)
            else:
                missed_skills.append(skill)
        skills_match_rate = len(matched_skills) / len(required_skills)
    else:
        skills_match_rate = 1.0
        
    skills_score = skills_match_rate * 100
    
    # 3. MiniLM Semantic Cosine Similarity (20%)
    try:
        resume_emb = model.encode([resume_text[:2500]]) # bound characters for CPU speed
        
        # Aggregate structural job descriptors for comparison context
        jd_summary = " ".join(
            jd_dict.get("keywords_critical", []) + 
            jd_dict.get("required_skills", []) + 
            jd_dict.get("tools_and_technologies", [])
        )
        if not jd_summary.strip():
            jd_summary = "General role description matching corporate requirements."
            
        jd_emb = model.encode([jd_summary])
        
        # Scikit-learn cosine similarity matrices dot-product
        semantic_sim = float(cosine_similarity(resume_emb, jd_emb)[0][0])
        semantic_score = max(0.0, semantic_sim) * 100
    except Exception as e:
        print(f"[ATS Scorer] Cosine similarity failed: {e}")
        semantic_score = 50.0 # safety average fallback
        
    # 4. Section Presence Check (10%)
    sections = {
        "experience": bool(re.search(r"\b(experience|employment|work history|career)\b", resume_lower)),
        "education": bool(re.search(r"\b(education|degree|university|college|academic)\b", resume_lower)),
        "skills": bool(re.search(r"\b(skills|technologies|tools|expertise|technical)\b", resume_lower)),
        "projects": bool(re.search(r"\b(projects|portfolio|accomplishments)\b", resume_lower))
    }
    section_match_rate = sum(sections.values()) / len(sections)
    section_score = section_match_rate * 100
    
    # 5. Metric/Number Density (10%)
    numbers = re.findall(r"\b\d+[%$₹kKmM]?\b", resume_text)
    # A standard ATS prefers high quantification. Full marks for >= 6 metrics
    quant_score = min(len(numbers) * 16.6, 100.0)
    
    # Complete Weighted Aggregation
    total_score = (
        keyword_score * 0.35 +
        skills_score  * 0.25 +
        semantic_score * 0.20 +
        section_score  * 0.10 +
        quant_score    * 0.10
    )
    
    # Classify Grading Tier
    total_score = round(total_score, 1)
    if total_score >= 90.0:
        grade = "A"
    elif total_score >= 75.0:
        grade = "B"
    elif total_score >= 60.0:
        grade = "C"
    else:
        grade = "D"
        
    return {
        "total_score": total_score,
        "breakdown": {
            "keywords": round(keyword_score, 1),
            "skills": round(skills_score, 1),
            "semantic": round(semantic_score, 1),
            "sections": round(section_score, 1),
            "quantification": round(quant_score, 1)
        },
        "missing_keywords": missed_critical,
        "missing_skills": missed_skills,
        "sections_found": sections,
        "quant_metrics_count": len(numbers),
        "grade": grade
    }
