from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import xgboost as xgb
import logging
from fastapi.responses import JSONResponse
from fastapi.requests import Request

#load trained model

model = joblib.load("../../models/xgboost_ranker_model.pkl")

#Define request body schema

class MSPRequest(BaseModel):
    job_id: int
    candidates: list[dict] #Each MSP's features

app=FastAPI(title="MSP Ranking API",description="Rank MSPs for a given job")

# ------------------------------------------------
# Configure logging
# ------------------------------------------------
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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
def rank_msp(request:MSPRequest):
    try:
        df=pd.DataFrame(request.candidates)
        features=["distance_km","price_quote","past_accept_rate","completion_rate","rating"]

        df["score"] = model.predict(df[features])

        ranked=df.sort_values('score',ascending=False).to_dict(orient='records')
        return {"job_id":request.job_id,"ranked_candidates":ranked}
    except Exception as e:
        logging.error(f"Error while ranking MSPs: {e}")
        raise
    