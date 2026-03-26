# ══════════════════════════════════════════════════════════════
#  MODEL.PY  (FIXED - Safe string handling)
# ══════════════════════════════════════════════════════════════
MODEL_PY = '''import os, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.DataFrame(); tfidf = TfidfVectorizer(); skill_vectors = None

def load():
    global df, tfidf, skill_vectors
    df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
    
    # Safe string conversion for all text columns
    text_cols = ["qualifications", "subjects", "programming_languages", 
                 "certifications", "interests", "description", "career_paths"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    
    df["combined"] = (
        df["qualifications"] + " " + df["qualifications"] + " " +
        df["subjects"] + " " + df["subjects"] + " " +
        df["programming_languages"] + " " +
        df["certifications"] + " " +
        df["interests"] + " " + df["interests"] + " " +
        df["description"] + " " +
        df["career_paths"]
    )
    
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000, sublinear_tf=True)
    skill_vectors = tfidf.fit_transform(df["combined"])
    print(f"[Model] {len(df)} skills loaded successfully")

def safe_str(val):
    """Safely convert any value to string and strip whitespace"""
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

def _ov(a, b):
    if not a or not b: 
        return []
    a_set = set(safe_str(a).lower().replace(",", " ").split())
    b_set = set(safe_str(b).lower().replace(",", " ").split())
    return [w.capitalize() for w in a_set & b_set if len(w) > 2][:4]

def _explain(skill, score, qual, langs, interests, subjects, certs, row):
    p = []
    if score >= 70:
        p.append(f"Based on a strong compatibility score of {score}%, {skill} stands out as an excellent match for your profile.")
    elif score >= 40:
        p.append(f"With a compatibility score of {score}%, {skill} is a well-suited choice aligned with your background.")
    else:
        p.append(f"{skill} has been identified as a potential growth direction with a compatibility score of {score}%.")

    if qual:
        m = _ov(qual, row["qualifications"])
        if m:
            p.append(f"Your educational background in {qual} directly aligns with the qualifications required for {skill}, especially in {', '.join(m)}.")
        else:
            p.append(f"Your educational background in {qual} aligns with the qualifications required for {skill}.")

    if subjects:
        m = _ov(subjects, row["subjects"])
        if m:
            p.append(f"Your knowledge of {', '.join(m)} is directly relevant to the academic demands of {skill}.")

    if langs:
        m = _ov(langs, row["programming_languages"])
        if m and safe_str(row["programming_languages"]) not in ["", "none"]:
            p.append(f"Your proficiency in {', '.join(m)} gives you a head start in {skill}.")

    if interests:
        m = _ov(interests, row["interests"])
        if m:
            p.append(f"Your passion for {', '.join(m)} strongly aligns with the work in {skill}.")

    if certs:
        m = _ov(certs, row["certifications"])
        if m and safe_str(row["certifications"]) not in ["", "none"]:
            p.append(f"Your certification(s) in {', '.join(m)} are recognised credentials in the {skill} domain.")

    sal = row["avg_salary"]
    if isinstance(sal, (int, float)) and not pd.isna(sal):
        sal_str = f"~${int(sal)//1000}K/yr"
    else:
        sal_str = "variable (based on business success)"

    p.append(f"Career-wise, {skill} opens doors to: {row['career_paths']}. Average salary: {sal_str} with {row['demand_level'].lower()} market demand.")
    return " ".join(p)

def recommend(qualification="", skills="", certifications="", interests="", programming_languages="", subjects="", top_n=5):
    global df, skill_vectors
    if df.empty or skill_vectors is None:
        load()

    user = " ".join(filter(None, [
        qualification, qualification,
        subjects, subjects,
        interests, interests,
        skills,
        certifications,
        programming_languages
    ]))

    if not user.strip():
        return []

    sims = cosine_similarity(tfidf.transform([user]), skill_vectors).flatten()
    results = []
    
    for rank, idx in enumerate(sims.argsort()[::-1][:top_n]):
        row = df.iloc[idx]
        score = round(float(sims[idx]) * 100, 1)
        
        results.append({
            "rank": rank + 1,
            "skill": row["skill"],
            "score": score,
            "description": row["description"],
            "career_paths": row["career_paths"],
            "avg_salary": row["avg_salary"],
            "demand_level": row["demand_level"],
            "explanation": _explain(row["skill"], score, qualification, programming_languages, 
                                   interests, subjects, certifications, row)
        })
    return results

load()
'''
