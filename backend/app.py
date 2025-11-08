from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from contextlib import asynccontextmanager
import sqlite3
from datetime import datetime
from typing import List

# Global variables
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
db_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and clean up at shutdown."""
    global tokenizer, model, db_conn

    checkpoint = "mr4/phobert-base-vi-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

    # initialize sqlite3 connection and ensure table exists
    db_conn = sqlite3.connect("sentiments.db", check_same_thread=False)
    db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    db_conn.commit()

    yield

    # cleanup
    if db_conn:
        db_conn.close()
        db_conn = None

    tokenizer = None
    model = None

app = FastAPI(
    title="Vietnamese Sentiment Analysis API",
    description="API phân tích cảm xúc tiếng Việt sử dụng PhoBERT",
    version="1.0.0",
    lifespan=lifespan
)

# Request model
class TextInput(BaseModel):
    text: str

# Response model
class SentimentResult(BaseModel):
    label: str
    score: float

class SentimentRecord(BaseModel):
    id: int
    text: str
    sentiment: str
    timestamp: str

@app.post("/predict", response_model=SentimentResult)
async def predict(input_data: TextInput):
    """Predict sentiment of input text"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = tokenizer(
            input_data.text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = predictions  # shape: (batch_size, num_labels)
            # get the predicted class index for the first item in the batch and ensure it's an int
            label_id = int(torch.argmax(probs, dim=-1)[0].item())
            label = model.config.id2label[label_id]
            score = float(probs[0, label_id].item())

        # store result in sqlite3 with timestamp format YYYY-MM-DD HH:MM:SS
        try:
            if db_conn:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                db_conn.execute(
                    "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
                    (input_data.text, label, ts)
                )
                db_conn.commit()
        except Exception:
            # don't fail the prediction if DB insert fails
            pass

        return SentimentResult(
            label=label,
            score=round(score, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/records", response_model=List[SentimentRecord])
async def get_records(limit: int = 50, offset: int = 0):
    """Return recent sentiment records ordered by timestamp desc with pagination."""
    if db_conn is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    try:
        cur = db_conn.cursor()
        cur.execute(
            "SELECT id, text, sentiment, timestamp FROM sentiments "
            "ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "text": r[1], "sentiment": r[2], "timestamp": r[3]}
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)