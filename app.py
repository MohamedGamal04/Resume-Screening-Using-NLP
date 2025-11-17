import streamlit as st
import pickle
import os
import io
import re
import pandas as pd
import numpy as np
import contractions
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Tuple

# NLP helpers (nltk + num2words)
import nltk
try:
    from num2words import num2words
except Exception:
    num2words = None
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Optional: PDF/DOCX reading
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    from docx import Document
except Exception:
    Document = None


@st.cache_resource
def load_model(model_name: str = "all-mpnet-base-v2"):
    from huggingface_hub import login
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        return SentenceTransformer(model_name, token=token) if token else SentenceTransformer(model_name)
    except TypeError:
        return SentenceTransformer(model_name, use_auth_token=token) if token else SentenceTransformer(model_name)


def load_saved_models(models_dir = "model\\"):
    job_embeddings = None
    resume_embeddings = None
    JOB_data = None
    Resume_data = None

    with open(os.path.join(models_dir, "job_embeddings.pkl"), 'rb') as f:
            job_embeddings = pickle.load(f)
    with open(os.path.join(models_dir, "resume_embeddings.pkl"), 'rb') as f:
            resume_embeddings = pickle.load(f)
    with open(os.path.join(models_dir, "job_data.pkl"), 'rb') as f:
            JOB_data = pickle.load(f)
    with open(os.path.join(models_dir, "resume_data.pkl"), 'rb') as f:
            Resume_data = pickle.load(f)
    return job_embeddings, resume_embeddings, JOB_data, Resume_data


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = contractions.fix(text)
    text = text.replace('.', ' . ')
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = "".join(num2words(int(word)) if word.isdigit() else word for word in text)
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.tag.pos_tag(text)
    lemmatized_words = []

    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_words)


def _parse_skill_list(value):
    if value is None:
        return []
    s = str(value)
    # split on common delimiters
    parts = re.split(r'[;,\/\n|]', s)
    parts = [p.strip().lower() for p in parts if p and len(p.strip()) > 0]
    # dedupe while preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def get_matched_skills(job_row=None, resume_text=None, job_text=None, resume_row=None):
    # prepare texts
    job_txt = '' if job_text is None else str(job_text)
    res_txt = '' if resume_text is None else str(resume_text)
    if job_row is not None:
        # try common job skills columns
        for col in ['skills', 'Skills', 'Skills_required', 'Skills Required', 'skills_required']:
            if col in job_row.index:
                job_skill_list = _parse_skill_list(job_row.get(col))
                break
        else:
            job_skill_list = []
        # try job preprocessed columns for fallback text
        if not job_txt:
            for col in ['Job_preprocessed_requirements', 'Job_requirements', 'Job Description', 'Job_Description', 'Job Description']:
                if col in job_row.index:
                    job_txt = str(job_row.get(col))
                    break
    else:
        job_skill_list = []

    if resume_row is not None:
        for col in ['skills', 'Skills', 'skills_list']:
            if col in resume_row.index:
                resume_skill_list = _parse_skill_list(resume_row.get(col))
                break
        else:
            resume_skill_list = []
        if not res_txt:
            for col in ['Resume_preprocessed_str', 'Resume_str']:
                if col in resume_row.index:
                    res_txt = str(resume_row.get(col))
                    break
    else:
        resume_skill_list = []

    # Normalize texts
    job_proc = preprocess_text(job_txt) if job_txt else ''
    res_proc = preprocess_text(res_txt) if res_txt else ''

    matched = []
    # If job skills present, check which appear in resume text
    if job_skill_list:
        for s in job_skill_list:
            if s and s in res_proc:
                matched.append(s)
    # Else if resume skills present, check which appear in job text
    if not matched and resume_skill_list:
        for s in resume_skill_list:
            if s and s in job_proc:
                matched.append(s)

    # Fallback: token intersection
    if not matched and job_proc and res_proc:
        job_tokens = set(job_proc.split())
        res_tokens = set(res_proc.split())
        common = job_tokens.intersection(res_tokens)
        # filter out short tokens
        matched = sorted([t for t in common if len(t) > 2])

    # return unique, limited list
    seen = set()
    out = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            out.append(m)
        if len(out) >= 20:
            break
    return out


def read_uploaded_file(file) -> str:
    name = file.name.lower()
    data = file.getvalue()
    if name.endswith('.txt'):
        return data.decode('utf-8', errors='ignore')
    if name.endswith('.pdf') and PyPDF2 is not None:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    if (name.endswith('.docx') or name.endswith('.doc')) and Document is not None:
        doc = Document(io.BytesIO(data))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    # fallback: try decode
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def encode_texts(model, texts: List[str]) -> List[np.ndarray]:
    # Model encode returns numpy arrays by default when convert_to_tensor=False
    embeddings = []
    for t in texts:
        clean = preprocess_text(t)
        embeddings.append(model.encode(clean, convert_to_tensor=False))
    return embeddings


def top_k_similarities(query_emb, embeddings, top_k=5) -> List[Tuple[int, float]]:
    q = torch.from_numpy(np.array(query_emb)).unsqueeze(0)
    embs = torch.from_numpy(np.array(embeddings))
    scores = util.pytorch_cos_sim(q, embs)[0]
    top_k_actual = min(top_k, scores.shape[0])
    top = torch.topk(scores, k=top_k_actual)
    indices = top.indices.cpu().numpy().tolist()
    values = top.values.cpu().numpy().tolist()
    return list(zip(indices, values))


def main():
    st.set_page_config(page_title="Resume ↔ Job Matcher", layout="wide")
    st.title("Resume Screening — Semantic Matching")

    st.sidebar.header("Settings")

    mode = st.sidebar.selectbox("Mode", [
        "Search jobs by resume",
        "Search resumes by job",
        "Match job with resume"
    ])

    # Use fixed default for models directory and allow selecting embedding model in the sidebar
    models_dir = "model"
    model_name = st.sidebar.selectbox(
        "Embedding model",
        [
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-mpnet-base-v2",
            "all-MiniLM-L6-v2",
        ],
        index=0,
    )
    top_k = st.sidebar.number_input("Top K results", min_value=1, max_value=50, value=5)

    # Load model (cached)
    with st.spinner("Loading embedding model..."):
        model = load_model(model_name)

    # Try to load saved embeddings/data (no UI shown if missing)
    job_embeddings, resume_embeddings, JOB_data, Resume_data = load_saved_models(models_dir)

    col1, col2 = st.columns(2)

    if mode == "Search jobs by resume":
        st.header("Search Jobs by Resume")
        uploaded = st.file_uploader("Upload a resume (txt/pdf/docx) or paste text below", type=['txt','pdf','docx'])
        resume_text = st.text_area("Or paste resume text here (overrides upload)")

        if uploaded and not resume_text:
            resume_text = read_uploaded_file(uploaded)

        if job_embeddings is None:
            # If job data pickle exists, build text and compute embeddings on demand
            if isinstance(JOB_data, pd.DataFrame) and not JOB_data.empty:
                # Build job requirement text
                JOB_data['Job_requirements'] = JOB_data.get('Job Title', '').astype(str) + '. ' + JOB_data.get('Job Description', '').astype(str)
                JOB_texts = JOB_data['Job_requirements'].astype(str).tolist()
                with st.spinner("Encoding job dataset from job_data.pkl... this may take a while"):
                    job_embeddings = encode_texts(model, JOB_texts)
            else:
                st.error("No job embeddings or job data found. Please place 'job_embeddings.pkl' and 'job_data.pkl' in the 'models' folder or create them using the notebook.")

        run = st.button("Find matching jobs")
        if run:
            if not resume_text:
                st.error("Please provide a resume (upload or paste text).")
            elif job_embeddings is None:
                st.error("No job embeddings available. Upload job CSV or save embeddings to 'models/'.")
            else:
                with st.spinner("Encoding resume and computing similarities..."):
                    try:
                        q_emb = model.encode(preprocess_text(resume_text), convert_to_tensor=False)
                    except Exception:
                        q_emb = model.encode(resume_text, convert_to_tensor=False)
                    top = top_k_similarities(q_emb, job_embeddings, top_k=top_k)
                    rows = []
                    for idx, score in top:
                        job_row = None
                        if isinstance(JOB_data, pd.DataFrame):
                            try:
                                job_row = JOB_data.iloc[idx]
                            except Exception:
                                job_row = None

                        def _safe_job_get(row, keys):
                            if row is None:
                                return ''
                            for k in keys:
                                if k in row.index:
                                    val = row.get(k)
                                    return '' if pd.isna(val) else str(val)
                            return ''

                        salary = _safe_job_get(job_row, ['Salary', 'salary', 'Salary Range', 'SalaryRange'])
                        experience = _safe_job_get(job_row, ['Experience', 'experience', 'Years Experience', 'YearsExperience'])
                        work_type = _safe_job_get(job_row, ['Work Type', 'work_type', 'work_type', 'Employment Type', 'employment_type'])
                        qualifications = _safe_job_get(job_row, ['Qualifications', 'qualifications', 'Requirements', 'requirements'])

                        # compute matched skills between this job and the input resume
                        try:
                            matched = get_matched_skills(job_row=job_row, resume_text=resume_text)
                        except Exception:
                            matched = []
                        matched_str = ', '.join(matched) if matched else ''

                        rows.append({
                            'job_index': idx,
                            'similarity': float(score),
                            'title': _safe_job_get(job_row, ['Job Title', 'Title', 'job_title']),
                            'company': _safe_job_get(job_row, ['Company', 'company', 'Employer']),
                            'salary': salary,
                            'experience': experience,
                            'work_type': work_type,
                            'qualifications': qualifications,
                            'matched_skills': matched_str,
                        })
                # Show top-K results (no minimum similarity filter)
                df = pd.DataFrame(rows)
                st.success(f"Found {len(df)} matches")
                titles = []
                for r in rows:
                    titles.append(r['title'])
                most_title = 0
                for t in titles:
                    if titles.count(t) > titles.count(most_title):
                        most_title = t
                st.success(f"Best match : {most_title} by %{(titles.count(most_title) * 100 / len(titles))}")
                if not df.empty:
                    st.dataframe(df[['job_index','title','company','similarity','salary','experience','work_type','qualifications','matched_skills']])

    elif mode == "Search resumes by job":
        st.header("Search Resumes by Job")
        uploaded = st.file_uploader("Upload a job description (txt/pdf/docx) or paste text below", type=['txt','pdf','docx'])
        job_text = st.text_area("Or paste job description here (overrides upload)")
        if uploaded and not job_text:
            job_text = read_uploaded_file(uploaded)

        if resume_embeddings is None:
            # If resume data pickle exists, compute embeddings from it
            if isinstance(Resume_data, pd.DataFrame) and not Resume_data.empty:
                resume_texts = Resume_data.get('Resume_str', Resume_data.iloc[:,0]).astype(str).tolist()
                with st.spinner("Encoding resume dataset from resume_data.pkl... this may take a while"):
                    resume_embeddings = encode_texts(model, resume_texts)
            else:
                st.error("No resume embeddings or resume data found. Please place 'resume_embeddings.pkl' and 'resume_data.pkl' in the 'models' folder or create them using the notebook.")

        run = st.button("Find matching resumes")
        if run:
            if not job_text:
                st.error("Please provide a job description (upload or paste text).")
            elif resume_embeddings is None:
                st.error("No resume embeddings available. Upload resume CSV or save embeddings to 'models/'.")
            else:
                with st.spinner("Encoding job and computing similarities..."):
                    try:
                        q_emb = model.encode(preprocess_text(job_text), convert_to_tensor=False)
                    except Exception:
                        q_emb = model.encode(job_text, convert_to_tensor=False)
                    top = top_k_similarities(q_emb, resume_embeddings, top_k=top_k)
                    rows = []
                    for idx, score in top:
                        res_row = None
                        if isinstance(Resume_data, pd.DataFrame):
                            try:
                                res_row = Resume_data.iloc[idx]
                            except Exception:
                                res_row = None
                        # compute matched skills between this resume and the input job
                        try:
                            matched = get_matched_skills(job_text=job_text, resume_row=res_row)
                        except Exception:
                            matched = []
                        matched_str = ', '.join(matched) if matched else ''

                        rows.append({
                            'resume_index': idx,
                            'similarity': float(score),
                            'preview': (str(res_row.get('Resume_str',''))[:250] if res_row is not None else ''),
                            'matched_skills': matched_str,
                        })
                # Show top-K results (no minimum similarity filter)
                df = pd.DataFrame(rows)
                st.success(f"Found {len(df)} matches")
                if not df.empty:
                    st.dataframe(df[['resume_index','similarity','preview','matched_skills']])

    else:
        st.header("Match Job with Resume (one-to-one)")
        col1, col2 = st.columns(2)
        with col1:
            upj = st.file_uploader("Upload job (txt/pdf/docx)", type=['txt','pdf','docx'], key='jobfile')
            job_text = st.text_area("Or paste job description here")
            if upj and not job_text:
                job_text = read_uploaded_file(upj)
        with col2:
            upr = st.file_uploader("Upload resume (txt/pdf/docx)", type=['txt','pdf','docx'], key='resfile')
            resume_text = st.text_area("Or paste resume here", key='resume_text')
            if upr and not resume_text:
                resume_text = read_uploaded_file(upr)

        run = st.button("Compute similarity")
        if run:
            if not job_text or not resume_text:
                st.error("Please provide both a job and a resume (upload or paste).")
            else:
                with st.spinner("Encoding and computing similarity..."):
                    try:
                        job_emb = model.encode(preprocess_text(job_text), convert_to_tensor=False)
                    except Exception:
                        job_emb = model.encode(job_text, convert_to_tensor=False)
                    try:
                        resume_emb = model.encode(preprocess_text(resume_text), convert_to_tensor=False)
                    except Exception:
                        resume_emb = model.encode(resume_text, convert_to_tensor=False)
                    score = util.pytorch_cos_sim(torch.from_numpy(np.array(job_emb)).unsqueeze(0),
                                                 torch.from_numpy(np.array(resume_emb)).unsqueeze(0))[0][0].item()
                    pct = score * 100
                    quality = 'Excellent' if pct >= 80 else 'Very Good' if pct >= 75 else 'Good' if pct >= 65 else 'Fair' if pct >= 50 else 'Poor'
                st.metric(label="Similarity", value=f"{pct:.1f}%", delta=quality)

if __name__ == '__main__':
    main()
