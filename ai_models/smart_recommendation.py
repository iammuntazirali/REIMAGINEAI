from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  


candidate_texts = [
    "Example comment 1 from platform",
    "Another forum post about AI and ML",
    "Feedback on toxic comment reduction",
    "How to improve online conversations",
    "General discussion about thread quality"
]


candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)


class RecommendationRequest(BaseModel):
    inputs: List[str]
    top_k: int = 5  


@app.post("/recommend")
def recommend(req: RecommendationRequest):
    results = []
    for input_text in req.inputs:
        query_emb = model.encode(input_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, candidate_embeddings)[0]
        top_results = cos_scores.topk(req.top_k)
        recommendations = [
            {"comment": candidate_texts[idx], "score": float(top_results.values[i])}
            for i, idx in enumerate(top_results.indices)
        ]
        results.append({"input": input_text, "recommendations": recommendations})
    return {"results": results}
