# XGBoost Learning-to-Rank API

A production-ready implementation of a **Learning-to-Rank (LTR)** model using **XGBoost**, served through a **FastAPI** microservice.  
This project simulates a real-world use case of ranking **Moving Service Providers (MSPs)** — logistics vendors who bid for moving jobs — and demonstrates how to train, evaluate, and deploy a ranking model using modern MLOps practices.

---

## 🧠 What This Project Is About

In logistics platforms (like movers, delivery, or service aggregators), customers post jobs and multiple vendors or **Moving Service Providers (MSPs)** respond with quotes.  
The system must **rank** these MSPs by their likelihood of providing the best outcome — considering factors like distance, reliability, price, and ratings.

This project uses **Learning-to-Rank (LTR)** machine learning to automate that process.

Instead of manually sorting vendors by heuristics (e.g., lowest price or closest distance), an ML model learns an **optimal ranking function** based on historical outcomes such as job acceptance, completion, and customer satisfaction.

---

## ⚙️ Features

- **Pairwise Learning-to-Rank** with `XGBRanker`
- **Feature Engineering** for realistic logistics data
- **Evaluation** using **NDCG@5** (Normalized Discounted Cumulative Gain)
- **Production-grade FastAPI Inference Service**
  - `/rank-msps` endpoint
  - Structured **logging**
  - **Pydantic** input validation
  - **Request/response middleware**
- Clean, modular code ready for cloud migration

---

## 🧩 How It Works

1. **Data Simulation**  
   A synthetic dataset of 50 job requests × 10 MSP candidates per job is created.  
   Each candidate has features like:
   - `distance_km` – distance to customer  
   - `price_quote` – quoted price for the move  
   - `past_accept_rate` – MSP’s previous acceptance rate  
   - `completion_rate` – fraction of completed jobs  
   - `rating` – average customer rating  

2. **Label Generation**  
   Historical outcomes are converted into **relevance labels** (0–3) indicating how good the match was.

3. **Model Training**  
   An `XGBRanker` is trained in **pairwise** mode to predict the optimal order of MSPs for each job.

4. **Evaluation**  
   The model is evaluated using **NDCG@5**, a standard ranking metric in search and recommendation systems.

5. **Serving**  
   The trained model is wrapped in a **FastAPI** endpoint:
   ```bash
   POST /rank-msps
---
It accepts a job and a list of MSP candidates, scores each one, and returns them sorted by predicted relevance.

---

## 🚀 Example API Call

**PowerShell:**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/rank-msps" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{
  "job_id": 101,
  "candidates": [
    {
      "msp_id": 1,
      "distance_km": 4.2,
      "price_quote": 1200,
      "past_accept_rate": 0.75,
      "completion_rate": 0.88,
      "rating": 4.5
    },
    {
      "msp_id": 2,
      "distance_km": 2.5,
      "price_quote": 1400,
      "past_accept_rate": 0.9,
      "completion_rate": 0.95,
      "rating": 4.7
    }
  ]
}'
```

**Response:**

```json
{
  "job_id": 101,
  "ranked_candidates": [
    {"msp_id": 2, "score": 7.21, "rating": 4.7, ...},
    {"msp_id": 1, "score": 6.95, "rating": 4.5, ...}
  ]
}
```

---

## 🧾 Folder Structure

```
xgboost-ranking-system/
├── src/
│   └── api/
│       └── main.py
├── xgboost_msp_ranking.ipynb
├── .gitignore
└── README.md
```

**Note:** The following are ignored by `.gitignore` and not shown above:
- `models/` - trained model files
- `venv/` - virtual environment
- `logs/` - log files
- `__pycache__/` - Python cache files

---

## 🧠 Technologies Used

| Category      | Technology                                      |
| ------------- | ----------------------------------------------- |
| ML Model      | XGBoost (`XGBRanker`)                           |
| Serving       | FastAPI + Uvicorn                               |
| Data Handling | Pandas, NumPy                                   |
| Evaluation    | NDCG metric                                     |
| Logging       | Python logging + middleware                     |
| Validation    | Pydantic                                        |
| Cloud-ready   | Google BigQuery + Vertex AI (planned extension) |

---

## 🧾 Setup Instructions

```bash
git clone https://github.com/<your-username>/xgboost-ranking-system.git
cd xgboost-ranking-system
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn xgboost pandas scikit-learn joblib
```

Run the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

Access Swagger Docs:

```
http://127.0.0.1:8000/docs
```

---

## 🧩 Author
Krish Batra

Agentic AI | Machine Learning | AI Automation

🌐 [My Digital Home](https://www.vybecode.in/)

---

## 🪄 License

MIT License © 2025 Krish Batra
