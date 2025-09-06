from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentiment_analysis import predict_sentiment
app= FastAPI(title="Sentiment Analysis API", description="API for sentiment analysis using Logistic Regression", version="1.0.0")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str

@app.post("/sentiment", response_model=SentimentResponse)

async def get_sentiment(request: SentimentRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    label = predict_sentiment(text)
    return SentimentResponse(sentiment=label)