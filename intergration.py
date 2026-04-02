import os
import re
import json
import csv
import threading
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning

import warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ----------------------------
# Paths & Storage Config
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "models", "category_model.pkl")
URGENCY_MODEL_PATH = os.path.join(BASE_DIR, "models", "urgency_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
RULE_KEYWORDS_PATH = os.path.join(BASE_DIR, "models", "urgency_keywords.json")

STORE_DIR = os.getenv("STORE_DIR", os.path.join(BASE_DIR, "data"))
STORE_PATH = os.path.join(STORE_DIR, "email_predictions.csv")
OUTBOX_DIR = os.path.join(BASE_DIR, "outbox")
STORE_LOCK = threading.Lock()

STORE_COLUMNS = [
    "timestamp", "source", "subject", "email_text",
    "predicted_category", "predicted_urgency", "technical_category"
]

CATEGORY_MAP = {
    "forum": "forum",
    "updates": "updates",
    "verify_code": "verify_code",
    "social_media": "social_media",
    "promotions": "promotions",
    "spam": "spam",
}

# ----------------------------
# Pydantic Models
# ----------------------------
class IngestEmailRequest(BaseModel):
    source: str = Field(default="Gmail")
    subject: str = Field(default="")
    body: str = Field(..., min_length=1)
    attachments: Optional[List[str]] = Field(default_factory=list)
    target_systems: Optional[List[str]] = Field(default_factory=list)
    callback_url: Optional[str] = None

class PredictionResponse(BaseModel):
    timestamp: str
    source: str
    subject: str
    predicted_category: str
    predicted_urgency: str
    technical_category: str
    routed_systems: List[str]
    callback_sent: bool

# ----------------------------
# Utility Functions
# ----------------------------
def clean_text(text: str) -> str:
    """Basic email cleaning."""
    text = "" if text is None else str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"Sent from my.*|Best regards.*|Thanks.*\n.*|Sincerely.*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip().lower()

def load_rule_keywords():
    """Load urgency keywords from JSON or return defaults."""
    if os.path.exists(RULE_KEYWORDS_PATH):
        with open(RULE_KEYWORDS_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("urgency_keywords", {})
    return {
        "high": ["urgent", "immediately", "critical", "asap"],
        "medium": ["request", "follow up", "update", "review"],
        "low": ["fyi", "newsletter", "digest"]
    }

def category_rule_fallback(text: str) -> str:
    """Fallback category based on keyword rules."""
    text = text.lower()
    if any(k in text for k in ["verification code", "otp", "password reset"]):
        return "verify_code"
    if any(k in text for k in ["click here", "scam", "unsubscribe"]):
        return "spam"
    if any(k in text for k in ["offer", "discount", "coupon"]):
        return "promotions"
    if any(k in text for k in ["like", "comment", "friend request"]):
        return "social_media"
    if any(k in text for k in ["forum", "thread", "discussion"]):
        return "forum"
    return "updates"

# ----------------------------
# Store / Logging Functions
# ----------------------------
def ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    if not os.path.exists(STORE_PATH):
        pd.DataFrame(columns=STORE_COLUMNS).to_csv(STORE_PATH, index=False)

def append_store(record: dict):
    ensure_store()
    with STORE_LOCK:
        file_exists = os.path.exists(STORE_PATH)
        with open(STORE_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STORE_COLUMNS)
            if not file_exists or os.path.getsize(STORE_PATH) == 0:
                writer.writeheader()
            writer.writerow({col: record.get(col) for col in STORE_COLUMNS})

def read_store_df() -> pd.DataFrame:
    ensure_store()
    try:
        with STORE_LOCK:
            return pd.read_csv(STORE_PATH)
    except Exception:
        return pd.DataFrame(columns=STORE_COLUMNS)

# ----------------------------
# Load Models
# ----------------------------
category_model = joblib.load(CATEGORY_MODEL_PATH) if os.path.exists(CATEGORY_MODEL_PATH) else None
urgency_model = joblib.load(URGENCY_MODEL_PATH) if os.path.exists(URGENCY_MODEL_PATH) else None
tfidf_vectorizer = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None
urgency_keywords = load_rule_keywords()

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Email Classifier API", version="1.0.0")

@app.get("/health")
def health():
    df = read_store_df()
    return {
        "status": "ok",
        "category_model_loaded": category_model is not None,
        "urgency_model_loaded": urgency_model is not None,
        "tfidf_loaded": tfidf_vectorizer is not None,
        "records": len(df)
    }

@app.get("/predictions")
def get_predictions(limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0)):
    df = read_store_df()
    if df.empty:
        return {"count": 0, "items": []}
    df_sorted = df.sort_values("timestamp", ascending=False)
    page_df = df_sorted.iloc[offset: offset + limit].where(pd.notnull(df_sorted), None)
    return {"count": int(len(df)), "items": page_df.to_dict(orient="records")}

@app.post("/ingest", response_model=PredictionResponse)
def ingest_email(payload: IngestEmailRequest):
    if not payload.body.strip():
        raise HTTPException(status_code=400, detail="Email body is required")

    text = clean_text(f"{payload.subject}. {payload.body}")

    # Category prediction
    if category_model:
        try:
            tech_cat = str(category_model.predict([text])[0]).lower()
        except Exception:
            tech_cat = category_rule_fallback(text)
    else:
        tech_cat = category_rule_fallback(text)
    predicted_category = tech_cat if tech_cat in CATEGORY_MAP else category_rule_fallback(text)

    # Urgency rule fallback (simplified)
    rule_label = "low"
    text_lower = text.lower()
    if any(k in text_lower for k in urgency_keywords.get("high", [])):
        rule_label = "high"
    elif any(k in text_lower for k in urgency_keywords.get("medium", [])):
        rule_label = "medium"

    predicted_urgency = rule_label  # fallback if ML model not available

    # Store results
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": payload.source,
        "subject": payload.subject or "(no subject)",
        "email_text": text,
        "predicted_category": predicted_category,
        "predicted_urgency": predicted_urgency,
        "technical_category": tech_cat
    }
    append_store(result)

    return {
        **result,
        "routed_systems": payload.target_systems,
        "callback_sent": False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)