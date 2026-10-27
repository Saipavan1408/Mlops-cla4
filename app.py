"""Task 5: Serve the saved pipeline (model.pkl) with FastAPI.
Run:  uvicorn app:app --reload      Docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load("model.pkl")     # same pipeline used by Tasks 2-4
    yield
    state.clear()


app = FastAPI(title="Titanic Survival API", version="1.0", lifespan=lifespan)


class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Ticket class (1, 2 or 3)")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, le=10)
    Parch: int = Field(..., ge=0, le=10)
    Fare: float = Field(..., ge=0)
    Embarked: Optional[Literal["C", "Q", "S"]] = None

    model_config = {"json_schema_extra": {"example": {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S"}}}


class Prediction(BaseModel):
    survived: int
    label: str
    probability_survived: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(p: Passenger):
    row = pd.DataFrame([p.model_dump()])
    model = state["model"]
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])
    return Prediction(survived=pred,
                      label="Survived" if pred else "Did not survive",
                      probability_survived=round(prob, 4))
