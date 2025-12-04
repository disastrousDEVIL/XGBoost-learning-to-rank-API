from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import xgboost as xgb
import logging
import os
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from time import time


#load trained model

model = joblib.load("../../models/xgboost_ranker_model.pkl")

# ------------------------------------------------
# Define request schemas with validation
# ------------------------------------------------
class MSPCandidate(BaseModel):
    msp_id: int
    distance_km: float = Field(..., ge=0, description="Distance to customer in km")
    price_quote: float = Field(..., gt=0, description="Quoted price for the move")
    past_accept_rate: float = Field(..., ge=0, le=1, description="MSP's past acceptance rate (0-1)")
    completion_rate: float = Field(..., ge=0, le=1, description="MSP's past completion rate (0-1)")
    rating: float = Field(..., ge=0, le=5, description="Average customer rating (0-5)")

class MSPRequest(BaseModel):
    job_id: int
    candidates: list[MSPCandidate]

app=FastAPI(title="MSP Ranking API",description="Rank MSPs for a given job")

# ------------------------------------------------
# Configure logging
# ------------------------------------------------
# Ensure logs directory exists (relative to this script's location)
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, "api.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------------------------------
# Middleware for logging requests and responses
# ------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time=time()
    try:
        logging.info(f"Incoming request: {request.method} {request.url.path}")
        response=await call_next(request)
        duration=round(time()-start_time,3)
        logging.info(f"Completed {request.method} {request.url.path} in {duration}s with status {response.status_code}")
        return response
    except Exception as e:
        duration = round(time() - start_time, 3)
        logging.error(f"Request {request.method} {request.url.path} failed in {duration}s: {e}")
        raise
    
# ------------------------------------------------
# Global exception handler
# ------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Error while processing {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "message": "Internal server error while ranking MSPs."},
    )

# ------------------------------------------------
# Inference endpoint
# ------------------------------------------------

@app.post("/rank-msps")
def rank_msp(request: MSPRequest):
    try:
        df = pd.DataFrame([c.dict() for c in request.candidates])
        features = ["distance_km", "price_quote", "past_accept_rate", "completion_rate", "rating"]

        df["score"] = model.predict(df[features])
        ranked = df.sort_values("score", ascending=False).to_dict(orient="records")

        logging.info(f"Successfully ranked {len(df)} MSPs for job {request.job_id}")
        return {"job_id": request.job_id, "ranked_candidates": ranked}

    except Exception as e:
        logging.error(f"Ranking failed for job {request.job_id}: {e}")
        raise
