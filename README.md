---
title: MedRecordAudit
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# MedRecordAudit — OpenEnv RL Environment

An RL environment where AI agents audit years of patient medical records to find missed diagnoses, dangerous drug interactions, contradictions between doctors, and declining lab trends that went unaddressed.

**Why this matters:** Medical errors from missed patterns are the 3rd leading cause of death in the US. Doctors get 15 minutes per patient — nobody reads 10 years of records. AI agents trained in this environment learn to strategically investigate medical histories and catch what humans miss.

## How It Works

```
Agent                          Environment (MedRecordAudit)
  |                                  |
  |---- reset(difficulty) ---------->|  Load patient case + task instructions
  |<--- state (task, patient, index) |
  |                                  |
  |---- step(read_record, id=5) --->|  Return full record content
  |<--- state + info ---------------|
  |                                  |
  |---- step(cross_reference) ----->|  Return drug/disease database info
  |<--- state + info ---------------|
  |                                  |
  |---- step(flag_issue) ---------->|  Record the finding
  |<--- state + info ---------------|
  |                                  |
  |---- step(submit_report) ------->|  End episode, compute final score
  |<--- reward (0.0 - 1.0) --------|
```

## 9 Tasks (3 Difficulty Levels)

### Easy (20 records, 15 step budget, 1 issue)
| Task | Title | What the agent must find |
|------|-------|------------------------|
| easy_001 | Allergy Safety Audit | Amoxicillin prescribed to penicillin-allergic patient |
| easy_002 | Anticoagulant Drug Interaction Check | Ibuprofen prescribed to warfarin patient |
| easy_003 | Medication Safety in Kidney Disease | Metformin continued despite eGFR < 30 |

### Medium (80 records, 25 step budget, 3 issues)
| Task | Title | What the agent must find |
|------|-------|------------------------|
| medium_001 | Polypharmacy Safety Review | SSRI hyponatremia + tramadol seizure risk + HbA1c trend |
| medium_002 | High-Risk Drug Monitoring Audit | Amiodarone thyroid monitoring gap + liver trend + renal decline |
| medium_003 | Missed Diagnosis & Malabsorption Audit | Missed celiac disease + omeprazole malabsorption + autoimmune screening gap |

### Hard (150 records, 30 step budget, 5-6 issues)
| Task | Title | What the agent must find |
|------|-------|------------------------|
| hard_001 | Complex Multi-System Safety Audit | 6 issues across 7 specialists (amiodarone-warfarin, metformin CKD, hyperkalemia, hyponatremia, steroids, PPI) |
| hard_002 | Autoimmune Disease Management Audit | 5 issues (HCQ eye screening, calcium-levothyroxine, INR gaps, nephritis trend, DEXA gap) |
| hard_003 | Extreme Polypharmacy Cascade Audit | 6 cascading issues (aspirin+warfarin, digoxin toxicity, glipizide hypoglycemia, zolpidem falls, ACE cough, PPI deficiencies) |

## State Space

On `reset()`, the agent receives:

```json
{
  "case_id": "easy_001",
  "difficulty": "easy",
  "task": {
    "task_id": "easy_001",
    "title": "Allergy Safety Audit",
    "instruction": "Audit this patient's records for ALLERGY SAFETY VIOLATIONS...",
    "focus_areas": ["allergy_violation"],
    "expected_findings": 1
  },
  "patient": {
    "id": "P-10042",
    "age": 45,
    "gender": "F",
    "known_conditions": ["asthma"],
    "current_medications": ["albuterol inhaler PRN"],
    "allergies": ["penicillin"],
    "chief_complaint": "sinus infection"
  },
  "records_available": 20,
  "record_index": [
    {"id": 1, "date": "2021-06-10", "type": "visit_note", "summary": "Initial allergy evaluation"},
    {"id": 2, "date": "2021-06-10", "type": "allergy_record", "summary": "Allergy documentation"},
    ...
  ],
  "budget_remaining": 15,
  "available_actions": ["read_record", "cross_reference", "flag_issue", "submit_report"]
}
```

The agent sees record summaries but NOT full content. It must choose which records to read within a limited budget.

## Action Space

| Action | Parameters | What it does | Budget cost |
|--------|-----------|-------------|-------------|
| `read_record` | `record_id: int` | Returns full content of a medical record | 1 step |
| `cross_reference` | `query: str` | Searches drug/disease/lab databases | 1 step |
| `flag_issue` | `type, description, evidence` | Flags a found problem | 1 step |
| `submit_report` | none | Ends episode, returns final score | 1 step |

### Issue types for `flag_issue`:
- `drug_interaction` — Two drugs that shouldn't be co-prescribed
- `drug_contraindication` — Drug given despite a condition that forbids it
- `allergy_violation` — Drug given despite documented allergy
- `declining_trend` — Lab values worsening over time without action
- `missed_monitoring` — Required tests/screenings not performed
- `contradiction` — Conflicting instructions from different doctors
- `missed_diagnosis` — Symptoms suggesting an uninvestigated condition

## Reward Structure (0.0 - 1.0)

```
findings_score (max 0.70):
  Correct critical finding:  +0.25
  Correct moderate finding:  +0.15
  Correct minor finding:     +0.10
  False positive:            -0.10

efficiency_bonus (max 0.15):
  = (budget_remaining / total_budget) x 0.15

completeness_bonus (max 0.15):
  = (correct_findings / total_issues) x 0.15

TOTAL = clamp(findings + efficiency + completeness, 0.0, 1.0)
```

Score is 0.0 when no findings are submitted.

## Dataset

- **9 patient cases** with realistic medical data
- **~750 total medical records** (visit notes, lab results, prescriptions)
- **29 hidden issues** based on real-world medical errors
- All drug names, lab values, and clinical language sourced from FDA, clinical guidelines, and medical literature

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns 200 |
| `POST` | `/reset` | Start new episode — body: `{difficulty, case_id?}` |
| `POST` | `/step` | Take an action — body: `{action, ...params}` |
| `GET` | `/state` | Get current state |

## Setup

### Local Development

```bash
pip install -r requirements.txt
uvicorn env.server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t medrecordaudit .
docker run -p 7860:7860 medrecordaudit
```

### Run Inference

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key"
export ENV_URL="http://localhost:7860"

# Run all 9 tasks (what judges run)
python inference.py

# Run a single task
python inference.py easy_001

# Run all tasks in a difficulty
python inference.py easy
```

### Run Tests

```bash
# All 58 tests
python -m pytest tests/ -v

# All 9 graders
python -m env.graders
```

## Environment Variables

```python
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be set
```

## Project Structure

```
MedRecordAudit/
├── inference.py              # Baseline LLM agent (judges run this)
├── Dockerfile                # Docker for HF Spaces
├── openenv.yaml              # OpenEnv config
├── pyproject.toml            # Python project config (required by openenv)
├── uv.lock                   # Dependency lock file (required by openenv)
├── requirements.txt          # pip dependencies
├── .gitignore                # Excludes __pycache__, etc.
├── README.md                 # This file
│
├── server/                   # Entry point for openenv deployment
│   └── app.py                # main() server entry point
│
├── env/                      # Environment code
│   ├── environment.py        # Core RL environment (reset/step/state)
│   ├── reward.py             # Reward computation (0.0-1.0)
│   ├── server.py             # FastAPI server (/reset, /step, /state, /health)
│   └── graders/              # 9 individual graders
│       ├── base.py           # Shared grader logic
│       ├── easy_001.py       # Allergy safety audit
│       ├── easy_002.py       # Warfarin drug interaction
│       ├── easy_003.py       # Metformin kidney safety
│       ├── medium_001.py     # Polypharmacy safety review
│       ├── medium_002.py     # High-risk drug monitoring
│       ├── medium_003.py     # Missed diagnosis audit
│       ├── hard_001.py       # Complex multi-system (6 issues)
│       ├── hard_002.py       # Autoimmune management (5 issues)
│       └── hard_003.py       # Extreme polypharmacy (6 issues)
│
├── tasks/                    # Task instructions per case
│   ├── easy_001/task.json    # "Allergy Safety Audit"
│   ├── easy_002/task.json    # "Anticoagulant Drug Interaction Check"
│   ├── easy_003/task.json    # "Medication Safety in Kidney Disease"
│   ├── medium_001/task.json  # "Polypharmacy Safety Review"
│   ├── medium_002/task.json  # "High-Risk Drug Monitoring Audit"
│   ├── medium_003/task.json  # "Missed Diagnosis & Malabsorption Audit"
│   ├── hard_001/task.json    # "Complex Multi-System Safety Audit"
│   ├── hard_002/task.json    # "Autoimmune Disease Management Audit"
│   └── hard_003/task.json    # "Extreme Polypharmacy Cascade Audit"
│
├── data/
│   ├── lab_ranges.json       # 50+ lab tests with reference ranges
│   ├── drugs.json            # 26 drugs with real interactions
│   ├── diseases.json         # 15 diseases with diagnostic criteria
│   └── cases/
│       ├── easy/             # 3 cases x 20 records
│       ├── medium/           # 3 cases x 80 records
│       └── hard/             # 3 cases x 150 records
│
└── tests/
    └── test_env.py           # 58 tests validating the environment
```

## Built For

**Meta PyTorch OpenEnv Hackathon x Scaler School of Technology** — India's first OpenEnv AI Hackathon in collaboration with Meta, Hugging Face, and PyTorch Foundation.
