# from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import json 
# import asyncio
# from typing import List , Dict
# import logging

# from extractive_summarizer import summarize
# from sentiment_analysis import predict_sentiment
# from toxicity import get_toxicity
# from semantic_search import semantic_search

# app = FastAPI(title="Community platform ML api", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # websocket connection
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket]= []
#     async def connect(self,websocket:WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket:WebSocket):
#         self.active_connections.remove(websocket)

#     async def broadcast(self, message:dict):
#         for connection in self.active_connections:
#             try:
#                 await connection.send_text(json.dumps(message))
#             except:
#                 await self.disconnect(connection)
# manager = ConnectionManager()

# class ContentRequest(BaseModel):
#     text: str
#     source:str= "reddit"
#     post_id: str= None

# class MLResponse(BaseModel):
#     sentiment :str
#     toxicity_score: float
#     toxicity_flagged: bool
#     summary: str= None
#     semantic_matches: List[dict] = []

# # Background task for processing

# async def process_content_background(content:str, post_id:str):
#     try:
#         sentiment = predict_sentiment(content)
#         toxicity_score= get_toxicity(content)
#         summary = summarize(content) if len(content) > 100 else content

#         result ={
#             "post_id":post_id,
#             "sentiment":sentiment,
#             "toxicity_score":toxicity_score,
#             "toxicity_flagged":toxicity_score> 0.6,
#             "summary":summary,
#             "processed_at": asyncio.get_event_loop().time()

#         }
#         await manager.broadcast({
#             "type":"ml_analyse",
#             "data":result
#         })
#     except Exception as e:
#         logging.error(f"Error processing content: {e}")

# @app.post("/analyze-content")
# async def analyze_content(request: ContentRequest, background_tasks: BackgroundTasks):
#     background_tasks.add_task(
#         process_content_background,
#         request.text,
#         request.post_id or "unknown"

#     )
#     return {"status":"processing","message":"analysis started"}

# @app.post("/sentiment")
# async def analyze_sentiment(request: ContentRequest):
#     sentiment = predict_sentiment(request.text)
#     return {"sentiment":sentiment, "post_id":request.post_id}

# @app.post("/toxicity")
# async def check_toxicity(request:ContentRequest):
#     score = get_toxicity(request.text)
#     return{
#         "toxicity_score":score,
#         "flagged":score>0.6,
#         "post_id":request.post_id
#     }

# @app.post("/summarize")
# async def create_summary(request: ContentRequest):
#     summary = summarize(request.text)
#     return {"summary":summary, "post_id": request.post_id}

# @app.post("/semantic-search")
# async def search_content(request:ContentRequest):
#     results=[]
#     return{"matches":results, "query":request.text}

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data= await websocket.receive_text()
#             message = json.loads(data)
#             if message.get("type")=="ping":
#                 await websocket.send_text(json.dumps({"type":"pong"}))

#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
# import os, json, asyncio, logging
# from typing import List, Dict
# from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from fastapi import APIRouter, Request
# from fastapi.responses import JSONResponse

# # Load env
# load_dotenv()
# FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "").split(",")

# # Import ML & Reddit modules
# from extractive_summarizer import summarize
# from sentiment_analysis import predict_sentiment
# from toxicity import get_toxicity
# from semantic_search import semantic_search
# from reddit_service import RedditService

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Community Platform ML API")

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=FRONTEND_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # WebSocket manager
# class ConnectionManager:
#     def __init__(self):
#         self.connections: List[WebSocket] = []
#     async def connect(self, ws: WebSocket):
#         await ws.accept(); self.connections.append(ws)
#     def disconnect(self, ws: WebSocket):
#         self.connections.remove(ws)
#     async def broadcast(self, msg: dict):
#         for ws in list(self.connections):
#             try: await ws.send_text(json.dumps(msg))
#             except: self.disconnect(ws)

# manager = ConnectionManager()
# reddit = RedditService()

# # Pydantic models
# class ContentRequest(BaseModel):
#     text: str
#     post_id: str = None

# # Background processing
# async def process_background(text: str, post_id: str):
#     try:
#         sentiment = predict_sentiment(text)
#         toxicity_score = get_toxicity(text)
#         summary = summarize(text) if len(text) > 100 else text
#         result = {
#             "post_id": post_id,
#             "sentiment": sentiment,
#             "toxicity_score": toxicity_score,
#             "toxicity_flagged": toxicity_score > 0.6,
#             "summary": summary
#         }
#         await manager.broadcast({"type":"ml_update","data":result})
#     except Exception as e:
#         logger.error(f"Background error: {e}")

# # Endpoints
# @app.post("/recommend")
# async def recommend(request: Request):
#     body = await request.json()
#     interests = body.get("inputs", [])
#     top_k = body.get("top_k", 5)
#     # Call your smart_recommendation.py model function here
#     from smart_recommendation import recommend_posts
#     results = recommend_posts(interests, top_k)
#     return {"recommended_items": results}


# @app.get("/")
# async def read_root():
#     return {"message": "Community Platform ML API is running 🚀"}


# @app.post("/analyze-content")
# async def analyze_content(req: ContentRequest, bg: BackgroundTasks):
#     bg.add_task(process_background, req.text, req.post_id or "")
#     return {"status":"started"}

# @app.get("/reddit/hot/{subreddit}")
# async def get_hot(subreddit: str, limit: int = 10):
#     try:
#         posts = reddit.get_hot_posts(subreddit, limit)
#         print(f"Fetched posts for /reddit/hot/{subreddit}?limit={limit}: {posts[:3]}")
#         return posts
#     except Exception as e:
#         print("Error in get_hot:", e)
#         raise HTTPException(400, str(e))


# @app.get("/reddit/search")
# async def search_reddit(query: str, subreddit: str = "all", limit: int = 25):
#     try:
#         return reddit.search_posts(query, subreddit, limit)
#     except Exception as e:
#         raise HTTPException(400, str(e))

# @app.websocket("/ws")
# async def ws_endpoint(ws: WebSocket):
#     await manager.connect(ws)
#     try:
#         while True:
#             msg = await ws.receive_text()
#             # handle ping/pong if needed
#     except WebSocketDisconnect:
#         manager.disconnect(ws)
import os, json, logging
from typing import List
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from semantic_search import semantic_search





# Load env
load_dotenv()
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "").split(",") or ["*"]

# Import ML & Reddit modules
from extractive_summarizer import summarize
from sentiment_analysis import predict_sentiment
from toxicity import get_toxicity
from semantic_search import semantic_search
from reddit_service import RedditService
from smart_recommendation import recommend_posts

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# app = FastAPI(title="Community Platform ML API")
app = FastAPI(title="reimagineai")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)
    async def broadcast(self, msg: dict):
        for ws in list(self.connections):
            try:
                await ws.send_text(json.dumps(msg))
            except:
                self.disconnect(ws)

manager = ConnectionManager()
reddit = RedditService()

# Pydantic models
class ContentRequest(BaseModel):
    text: str
    post_id: str = None

class SentimentRequest(BaseModel):
    text: str

class ToxicityRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str

class SemanticSearchRequest(BaseModel):
    text: str

# Background processing
async def process_background(text: str, post_id: str):
    try:
        sentiment = predict_sentiment(text)
        toxicity_score = get_toxicity(text)
        summary = summarize(text) if len(text) > 100 else text
        result = {
            "post_id": post_id,
            "sentiment": sentiment,
            "toxicity_score": toxicity_score,
            "toxicity_flagged": toxicity_score > 0.6,
            "summary": summary
        }
        await manager.broadcast({"type": "ml_update", "data": result})
    except Exception as e:
        logger.error(f"Background error: {e}")

# Endpoints

@app.post("/recommend")
async def recommend(request: Request):
    body = await request.json()
    interests = body.get("inputs", [])
    top_k = body.get("top_k", 5)
    results = recommend_posts(interests, top_k)
    return {"recommended_items": results}

@app.get("/")
async def read_root():
    return {"message": "Community Platform ML API is running 🚀"}

@app.post("/analyze-content")
async def analyze_content(req: ContentRequest, bg: BackgroundTasks):
    bg.add_task(process_background, req.text, req.post_id or "")
    return {"status": "started"}

# @app.get("/reddit/hot/{subreddit}")
# async def get_hot(subreddit: str, limit: int = 10):
#     try:
#         posts = reddit.get_hot_posts(subreddit, limit)
#         print(f"Fetched posts for /reddit/hot/{subreddit}?limit={limit}: {posts[:3]}")
#         return posts
#     except Exception as e:
#         print("Error in get_hot:", e)
#         raise HTTPException(400, str(e))
@app.get("/reddit/hot/{subreddit}")
async def hot_posts(subreddit: str, limit: int = 10):
    try:
        return reddit.get_hot_posts(subreddit, limit)
    except Exception as e:
        raise HTTPException(400, str(e))
    
@app.get("/reddit/post/{post_id}")
async def post_details(post_id: str):
    try:
        return reddit.get_post_with_comments(post_id)
    except Exception as e:
        raise HTTPException(400, str(e))



@app.get("/reddit/hot-detailed/{subreddit}")
async def reddit_hot_detailed(subreddit: str, limit: int = 10):
    try:
        posts = reddit.get_detailed_hot_posts(subreddit, limit)
        return posts
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.get("/reddit/search")
async def search_reddit(query: str, subreddit: str = "all", limit: int = 25):
    try:
        return reddit.search_posts(query, subreddit, limit)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/sentiment")
async def sentiment(req: SentimentRequest):
    try:
        result = predict_sentiment(req.text)
        return {"sentiment": result}
    except Exception as e:
        raise HTTPException(500, f"Sentiment analysis error: {e}")

@app.post("/toxicity")
async def toxicity(req: ToxicityRequest):
    try:
        score = get_toxicity(req.text)
        return {"toxicity_score": float(score)}
    except Exception as e:
        raise HTTPException(500, f"Toxicity analysis error: {e}")

@app.post("/summarize")
async def summarize_endpoint(req: SummarizeRequest):
    try:
        summary_text = summarize(req.text)
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(500, f"Summarization error: {e}")

@app.post("/semantic-search")
async def semantic_search_endpoint(req: SemanticSearchRequest):
    try:
        matches = semantic_search(req.text)
        # matches should be [{"text": "...", "score": 0.87}, ...]
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(500, f"Semantic search error: {e}")

# @app.post("/semantic-search")
# async def semantic_search_endpoint(req: SemanticSearchRequest):
#     try:
#         matches = semantic_search(
#             req.text,
#             model,
#             word_to_idx,
#             candidate_texts,
#             candidate_embs,
#             device,
#         )
#         return {"matches": matches}
#     except Exception as e:
#         raise HTTPException(500, f"Semantic search error: {e}")
# matches = semantic_search(req.text)
# @app.post("/semantic-search")
# async def semantic_search_endpoint(req: SemanticSearchRequest):
#     try:
#         matches = semantic_search(
#             req.text,
#             model,
#             word_idx,
#             candidate_texts,
#             candidate_embs,
#             device,
#         )
#         return {"matches": matches}
#     except Exception as e:
#         raise HTTPException(500, f"Semantic search error: {e}")





@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()

            # Handle ping/pong messages
            if msg.lower() == "ping":
                await ws.send_text("pong")
            elif msg.lower() == "pong":
                # client responded to our ping, no action needed
                pass
            else:
                # Broadcast or echo the message
                await manager.broadcast(f"Client says: {msg}")

    except WebSocketDisconnect:
        manager.disconnect(ws)
        await manager.broadcast("A client disconnected")