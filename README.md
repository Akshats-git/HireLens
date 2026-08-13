# HireLens

AI-powered resume-to-job-description matching system. Scores resumes against job descriptions using semantic embeddings and NER-based skill extraction, with dual views for candidates (single resume analysis) and recruiters (bulk ranking + filtering).

Live deployment: **http://13.51.207.145** (AWS EC2 eu-north-1)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client (Browser)                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP :80
                    ┌───────────▼───────────┐
                    │    nginx (port 80)     │
                    │   reverse proxy       │
                    └──────┬──────────┬─────┘
                           │          │
              / (SPA)      │          │  /api/*  /health
                           │          │
              ┌────────────▼──┐  ┌────▼──────────────────┐
              │   React +     │  │   FastAPI backend      │
              │   Vite SPA    │  │   (port 8000)          │
              │   (port 3000) │  │                        │
              │               │  │  ┌──────────────────┐  │
              │  ┌──────────┐ │  │  │  ML Pipeline     │  │
              │  │Candidate │ │  │  │                  │  │
              │  │  View    │ │  │  │  NERExtractor    │  │
              │  └──────────┘ │  │  │  (spaCy trf)    │  │
              │  ┌──────────┐ │  │  │       ↓          │  │
              │  │Recruiter │ │  │  │  EmbeddingModel  │  │
              │  │  View    │ │  │  │  (MiniLM-L6)    │  │
              └──────────────┘  │  │       ↓          │  │
                                │  │  MatchScorer     │  │
                                │  │  (4-component)   │  │
                                │  └──────────────────┘  │
                                │          │              │
                                │  ┌───────▼──────┐       │
                                │  │  Redis cache │       │
                                │  │  (port 6379) │       │
                                │  └──────────────┘       │
                                │  ┌───────────────┐      │
                                │  │  PostgreSQL   │      │
                                │  │  (port 5432)  │      │
                                │  └───────────────┘      │
                                └────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │      AWS S3             │
                              │  hirelens-models-*      │  ← fine-tuned weights
                              │  hirelens-resumes-*     │  ← uploaded PDFs
                              └─────────────────────────┘

CI/CD (GitHub Actions → ghcr.io → EC2)
──────────────────────────────────────
push to main
    │
    ├── Job 1: Backend lint (black + flake8) + pytest (30 tests)
    ├── Job 2: Frontend lint (ESLint) + Vite production build
    ├── Job 3: Build & push Docker images → ghcr.io
    └── Job 4: SSH deploy to EC2 → docker compose pull + rolling restart
```

---

## Achieved Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| NDCG@10 | **0.962** | 0.80 | ✓ |
| Precision@1 | **0.896** | — | — |
| AUC-ROC | **0.837** | 0.80 | ✓ |
| MRR | **0.948** | — | — |
| NER F1 (skill extraction) | **0.771** | 0.88 | — |
| Scoring latency (GPU, single) | **165 ms** | <500 ms | ✓ |
| Bulk 50 resumes | **8.1 s** (162 ms/resume) | — | — |
| Model size | **608 MB** | — | — |
| Test suite | **30/30 passing** | — | ✓ |

> Evaluation on 712 held-out resume–job pairs. Latency measured on GTX 1650; AWS t3.micro (CPU-only) is ~3–5× slower.

---

## Datasets

### 1. Raw Sources

| Dataset | Rows | Used For |
|---------|------|----------|
| Kaggle Resume Dataset | 19,020 resumes | Training pairs + NER eval |
| LinkedIn Job Postings | 3.3M listings | Job description corpus |
| Kaggle Structured Resumes | 1,200 resumes | NER F1 evaluation |

**Resumes** (`data/raw/resumes_clean.csv`): scraped and cleaned resume text spanning engineering, data science, finance, marketing, and healthcare roles.

**LinkedIn Jobs** (`data/raw/linkedin_jobs/`): 3.3M job postings with structured metadata — skills, industries, salaries, company details. Filtered to English-language postings with at minimum a job description and skills list.

### 2. Processing Pipeline

```
Raw CSVs
    │
    ▼  src/data/ingestion.py
Download from Kaggle, normalize columns, drop rows under 50 chars
    │
    ▼  src/data/preprocessing.py
Clean text (NFKC, strip PII and decorative rules), detect resume sections,
then build labelled pairs by weak supervision:
  - Positive (1.0): posting matches one of the resume category's keywords
  - Negative (0.0): posting matches none of them — a different domain,
    not random noise
  - Ratio: 1 positive to 2 negatives per resume
    │
    ▼
data/processed/train_pairs.csv   (5,692 pairs — 1,897 pos / 3,795 neg)
data/processed/val_pairs.csv     (712 pairs — 238 pos / 474 neg)
data/processed/test_pairs.csv    (held out for evaluation)

Separately, src/data/eval_dataset.py joins a structured resume dataset against
real postings and scores each pair with the same four rules the API uses,
producing data/eval/eval_pairs.csv as a model-independent reference set.
```

### 3. Model Training

Fine-tuned `sentence-transformers/all-MiniLM-L6-v2` on the generated pairs using `CosineSimilarityLoss`:

```
Base model : sentence-transformers/all-MiniLM-L6-v2 (384-dim)
Epochs     : 5
Batch size : 32
LR         : 2e-5 (cosine schedule, 200 warmup steps)
FP16       : enabled
Device     : NVIDIA GTX 1650 (3.9 GB VRAM)
Duration   : ~56 minutes

Val Pearson cosine  : 0.622  (vs 0.355 for base model)
Val Spearman cosine : 0.561
```

The fine-tuned model improves Pearson correlation by **+75%** relative to the off-the-shelf base model on the domain-specific resume/JD pairs.

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **ML / NLP** | sentence-transformers | 3.0.1 |
| | PyTorch | 2.3.1 |
| | spaCy (en_core_web_trf) | 3.7.5 |
| | scikit-learn | 1.5.1 |
| | HuggingFace Transformers | 4.42.4 |
| **Backend** | FastAPI | 0.111.1 |
| | Uvicorn | 0.30.3 |
| | Pydantic v2 | 2.8.2 |
| | SQLAlchemy (async) | 2.0.31 |
| | asyncpg | 0.29.0 |
| **Frontend** | React | 18 |
| | Vite | 5 |
| | Tailwind CSS | 3 |
| **Data** | PostgreSQL | 16.3 |
| | Redis | 7.2 |
| **PDF** | pdfplumber | 0.11.2 |
| | PyMuPDF | 1.24.7 |
| **Experiment tracking** | MLflow | 2.17.2 |
| **Infrastructure** | Docker + Docker Compose | — |
| | nginx | alpine |
| | AWS EC2 (t3.micro) | eu-north-1 |
| | AWS S3 | — |
| | GitHub Actions | — |

---

## Local Setup

### Prerequisites
- Python 3.12
- Node.js 20+
- Docker + Docker Compose
- Git

### 1. Clone and install

```bash
git clone https://github.com/Akshats-git/HireLens.git
cd HireLens

# Python environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# spaCy models (not on PyPI — install from GitHub releases)
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl"
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

# Frontend
cd frontend && npm ci && cd ..
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env — minimum required keys:
#   POSTGRES_PASSWORD=<any password>
#   APP_SECRET_KEY=<random 32-char hex>
```

See [Environment Variables](#environment-variables) for the full list.

### 3. Start services

```bash
# Start postgres + redis
docker compose up -d postgres redis

# Run backend (from repo root)
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Run frontend (in another terminal)
cd frontend && npm run dev
```

Open **http://localhost:5173** — the React dev server proxies `/api/` to `:8000`.

### 4. Run tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

### 5. Train model (optional — pre-trained weights included)

```bash
# Generate training pairs from raw data first
python -m src.data.data_pipeline

# Fine-tune
python -m src.models.train --epochs 5 --batch-size 32

# Evaluate
python -m src.evaluation.metrics
```

---

## AWS Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- EC2 key pair at `~/.ssh/hirelens.pem`
- GitHub repo secrets set (see below)

### GitHub Secrets Required

| Secret | Value |
|--------|-------|
| `EC2_HOST` | EC2 public IP (e.g. `13.51.207.145`) |
| `EC2_SSH_KEY` | Contents of your `.pem` key file |
| `POSTGRES_PASSWORD` | Strong password for Postgres |

> `GITHUB_TOKEN` is provided automatically — no setup needed for ghcr.io push.

### First-time EC2 bootstrap

```bash
# Launch Ubuntu 22.04 LTS t3.micro in eu-north-1
# Add inbound rules: SSH (22), HTTP (80)

# Bootstrap (installs Docker, nginx, swap, CloudWatch agent)
EC2_HOST=<your-ip> \
SSH_KEY=~/.ssh/hirelens.pem \
GHCR_USER=<github-username> \
GHCR_TOKEN=<github-pat> \
POSTGRES_PASSWORD=<password> \
bash aws/deploy.sh --bootstrap
```

### Automatic deploys

Every push to `main` triggers the CI/CD pipeline:

1. Python lint + 30 unit tests
2. Frontend ESLint + Vite build check
3. Build multi-stage Docker images and push to `ghcr.io`
4. SSH into EC2, pull new images, rolling restart (postgres/redis kept running), smoke test

No manual steps needed after the initial bootstrap.

### Manual re-deploy

```bash
EC2_HOST=13.51.207.145 \
SSH_KEY=~/.ssh/hirelens.pem \
GHCR_USER=<github-username> \
GHCR_TOKEN=<github-pat> \
POSTGRES_PASSWORD=<password> \
bash aws/deploy.sh
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_PASSWORD` | Yes | — | PostgreSQL password |
| `DATABASE_URL` | Yes | — | Full async DB URL (`postgresql+asyncpg://...`) |
| `REDIS_URL` | No | in-memory | Redis connection string |
| `APP_SECRET_KEY` | Yes | — | 32-byte hex secret for JWT signing |
| `JWT_SECRET_KEY` | Yes | — | 32-byte hex secret for tokens |
| `API_WORKERS` | No | `4` | Uvicorn worker count (`1` on t3.micro) |
| `API_RELOAD` | No | `false` | Hot-reload (dev only) |
| `LOG_LEVEL` | No | `INFO` | Loguru log level |
| `APP_ENV` | No | `development` | `production` disables debug features |
| `SPACY_MODEL` | No | `en_core_web_sm` | spaCy model (`en_core_web_trf` for best accuracy) |
| `MODEL_CACHE_DIR` | No | `models/cache` | Directory for embedding cache |
| `FINE_TUNED_MODEL_PATH` | No | `models/fine_tuned/hirelens_matcher` | Path to fine-tuned sentence-transformer |
| `AWS_DEFAULT_REGION` | No | `us-east-1` | AWS region for S3 |
| `AWS_S3_RESUMES_BUCKET` | No | — | S3 bucket for uploaded resumes |
| `AWS_S3_MODELS_BUCKET` | No | — | S3 bucket for model artifacts |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `MLFLOW_TRACKING_URI` | No | — | MLflow server URL for experiment logging |

> **Never commit credentials.** `.env` and `.env.prod` are in `.gitignore`. AWS credentials live in `~/.aws/credentials`, not in any project file.

---

## API Reference

### Candidate

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/candidate/analyze` | Score one resume PDF against a job description |

**Request:** `multipart/form-data` — `resume` (PDF ≤ 10 MB) + `job_description` (string ≥ 50 chars)

**Response:**
```json
{
  "score_pct": 82.5,
  "label": "Good",
  "matched_skills": ["python", "docker", "aws"],
  "missing_skills": ["kubernetes", "terraform"],
  "suggestions": ["Add Kubernetes experience to match senior DevOps requirements"],
  "processing_time_ms": 347
}
```

### Recruiter

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/recruiter/bulk-analyze` | Score up to 50 resumes, returns ranked list |
| `GET` | `/api/recruiter/candidate/{id}` | Full analysis for one candidate |
| `POST` | `/api/recruiter/filter` | Filter/re-rank by score, skills, experience level |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status":"healthy","models_loaded":true}` |
| `GET` | `/docs` | Swagger UI (all endpoints) |

---

## Project Structure

```
HireLens/
├── backend/                  # FastAPI application
│   ├── main.py               # App factory, lifespan, CORS
│   ├── routers/
│   │   ├── candidate.py      # /api/candidate/analyze
│   │   └── recruiter.py      # /api/recruiter/*
│   ├── services/
│   │   ├── ml_service.py     # Singleton: NER + embeddings + scorer
│   │   ├── cache_service.py  # Redis with in-memory fallback
│   │   └── pdf_service.py    # pdfplumber + PyMuPDF extraction
│   └── schemas/              # Pydantic request/response models
├── src/
│   ├── models/
│   │   ├── train.py          # Fine-tuning script
│   │   └── scorer.py         # 4-component MatchScorer
│   ├── features/
│   │   ├── ner.py            # spaCy NER + taxonomy skill extraction
│   │   └── embeddings.py     # Sentence-transformer wrapper + cache
│   ├── evaluation/
│   │   └── metrics.py        # NDCG, MRR, AUC-ROC, NER F1
│   ├── monitoring/
│   │   └── monitor.py        # Metric-regression check + retraining trigger
│   └── data/
│       ├── ingestion.py      # Raw CSV loading + cleaning
│       ├── preprocessing.py  # Text normalization
│       ├── eval_dataset.py   # Rule-scored held-out evaluation set
│       └── data_pipeline.py  # Stage orchestrator
├── frontend/                 # React + Vite + Tailwind
│   └── src/pages/
│       ├── Landing.jsx
│       ├── Candidate.jsx     # Single resume upload + score display
│       └── Recruiter.jsx     # Bulk upload + ranked table + filters
├── data/
│   ├── raw/                  # Original datasets (git-ignored)
│   ├── processed/            # train/val pairs (git-ignored)
│   ├── eval/                 # Rule-scored eval pairs (git-ignored)
│   └── skills_taxonomy.json  # 595 tech + 51 soft + 35 cert skills
├── models/
│   ├── fine_tuned/hirelens_matcher/   # 608 MB fine-tuned weights
│   └── cache/embeddings/             # SHA-256 keyed embedding cache
├── docker/
│   ├── Dockerfile.backend    # Multi-stage Python image
│   ├── Dockerfile.frontend   # Node builder → nginx static
│   └── postgres/init.sql     # DB + extension setup
├── aws/
│   ├── deploy.sh             # Local → EC2 deploy script
│   ├── setup-ec2.sh          # Bootstrap (Docker, nginx, swap)
│   ├── s3-setup.sh           # S3 bucket creation with encryption
│   └── cloudwatch-agent.json # CloudWatch log/metric config
├── tests/
│   ├── integration/          # FastAPI endpoint tests (30 total)
│   └── conftest.py           # Fixtures with mocked ML service
├── .github/workflows/
│   └── ci-cd.yml             # 4-job CI/CD pipeline
├── docs/
│   └── metrics_report.json   # Full evaluation + latency report
├── docker-compose.yml        # Local dev (postgres + redis)
├── docker-compose.prod.yml   # Production (all 4 services)
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Test + lint dependencies
├── dvc.yaml                  # DVC stage definitions
└── configs/config.yaml       # Master configuration (also the DVC params file)
```

---

## Screenshots

> _Add screenshots here after the UI is finalized._

| View | Description |
|------|-------------|
| Landing page | Role selection (Candidate / Recruiter) |
| Candidate view | Upload resume PDF + paste job description → score breakdown with matched/missing skills |
| Recruiter view | Upload up to 50 resumes → ranked table with score badges, filter by score/skills/experience |

---

## License

MIT
