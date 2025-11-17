# Resume Screening Using NLP

Semantic search app to match resumes and job descriptions using Sentence-Transformers and cosine similarity. Includes a Streamlit UI and a notebook for data prep, profiling, and embedding generation.

## Features
- Semantic matching: resumes ↔ jobs via transformer embeddings
- Top-K results with similarity scores
- Skills overlap extraction (simple NLP heuristic)
- Preprocessing pipeline (NLTK, contractions, lemmatization)
- Persisted embeddings/data as pickles under `model/`
- Streamlit UI with model selection and token-gated access

## Project Structure
- `app.py`: Streamlit UI for searching and one-to-one matching
- `ResumeScreeningUsingNLP.ipynb`: Notebook for data loading, preprocessing, profiling, and embedding generation
- `requirements.txt`: Python dependencies
- `model/`: Saved artifacts (`job_embeddings.pkl`, `resume_embeddings.pkl`, `job_data.pkl`, `resume_data.pkl`)
- `reports/`: Data profiling HTML outputs

## Prerequisites
- Python 3.10+ recommended (Windows + conda works well)
- Hugging Face account and API token (required by the app)

## Setup (Conda, Windows PowerShell)
```powershell
# Create and activate environment (adjust Python version if needed)
conda create -n torch python=3.10 -y
conda activate torch

# Install project dependencies
pip install -r requirements.txt

# Download NLTK data used in preprocessing
python -c "import nltk; [nltk.download(p) for p in ['punkt','averaged_perceptron_tagger','wordnet','omw-1.4','stopwords']]"
```

## Hugging Face Token
- Create a token: https://huggingface.co/settings/tokens (read access is enough)
- The app will ask for the token in the sidebar before loading models

Optionally, you can login once via CLI (not required by the app):
```powershell
pip install huggingface_hub
huggingface-cli login
```

## Generate Embeddings (Notebook)
If you don’t already have pickles in `model/`, open the notebook and run the embedding cells:
- File: `ResumeScreeningUsingNLP.ipynb`
- It saves: `model/job_embeddings.pkl`, `model/resume_embeddings.pkl`, `model/job_data.pkl`, `model/resume_data.pkl`

## Run the App
```powershell
streamlit run app.py
```
- Enter your Hugging Face token in the sidebar
- Choose a model and Top-K value
- Use one of the three modes: 
  - Search jobs by resume
  - Search resumes by job
  - Match job with resume (one-to-one)

## Troubleshooting
- Missing packages or DLL load errors on Windows:
  - Ensure Microsoft Visual C++ Redistributable is installed
  - Reinstall/upgrade key libs: `pip install -U sentence-transformers transformers huggingface-hub`
  - Install PyTorch following official instructions for your system: https://pytorch.org/get-started/locally/
- Hugging Face auth/model download errors:
  - Verify your token and internet access
  - For offline use, pre-download a model and set the model selector to a local path
- NLTK resource not found: rerun the NLTK download command above

## Notes
- Default models include: `all-mpnet-base-v2`, `paraphrase-MiniLM-L6-v2`, `paraphrase-mpnet-base-v2`, `all-MiniLM-L6-v2`
- App requires token entry even for public models to keep behavior consistent
- Profiles are saved to `reports/` and the app/notebook creates the folder if missing

## License
For internal or educational use. Review dataset licenses from Kaggle/Hugging Face before redistribution.
