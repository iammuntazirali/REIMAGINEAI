from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# api.py
from extractive_summarizer import summarize



class SummarizeRequest(BaseModel):
    text: str
    method: str = "abstractive"
    max_length: int = 150

class SummarizeResponse(BaseModel):
    summary: str

app = FastAPI(title="Text Summarization API")

@app.get("/")
def home():
    return {"message": "Server running successfully 🚀"}


@app.post("/summarize", response_model=SummarizeResponse)

async def get_summary(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    result = summarize(req.text, method= req.method, max_length=req.max_length)
    return SummarizeResponse(summary=result)