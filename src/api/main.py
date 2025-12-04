from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import xgboost as xgb

#load trained model

model = joblib.load("../../models/xgboost_ranker_model.pkl")

#Define request body schema

class MSPRequest(BaseModel):
    job_id: int
    candidates: list[dict] #Each MSP's features

app=FastAPI(title="MSP Ranking API",description="Rank MSPs for a given job")

#Inference endpoint

@app.post("/rank-msps")
def rank_msp(request:MSPRequest):
    df=pd.DataFrame(request.candidates)
    features=["distance_km","price_quote","past_accept_rate","completion_rate","rating"]

    df["score"] = model.predict(df[features])

    ranked=df.sort_values('score',ascending=False).to_dict(orient='records')
    return {"job_id":request.job_id,"ranked_candidates":ranked}