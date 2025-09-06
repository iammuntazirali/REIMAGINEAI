from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from typing import List

app = FastAPI()
api_keys = ['AIzaSyBGoP2YHmhsZVXLjg3PCMaHTljykxA9Bls']

current_key_index = 0

cache = {}

class QuotaExceededError(Exception):
    pass

def call_api_keys(text, api_key):


    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 429:
      
        raise QuotaExceededError("API quota exceeded")
    
    elif response.status_code != 200:
        
        raise Exception(f"API returned status code {response.status_code}: {response.text}")
    
    result = response.json()
    toxicity_score = result['attributeScores']['TOXICITY']['summaryScore']['value']
    return toxicity_score


def get_toxicity(text):

    global current_key_index
    if text in cache:
        return cache.pop(text)
    
    while current_key_index < len(api_keys):
        try:
            score = call_api_keys(text, api_keys[current_key_index])
            cache[text] = score
            return score

        except QuotaExceededError:
            print(f"Quota exceeded for API key index {current_key_index}, switching to next key.")
            current_key_index += 1

    raise Exception("All API keys quota exhausted")


class ThreadRequest(BaseModel):
    text: str  

threshold = 0.6

@app.post("/check_thread_toxicity/")
def check_thread_toxicity(req: ThreadRequest):
    comments = [line.strip() for line in req.text.split('\n') if line.strip()]

    mid_toxicity_comments = []
    high_toxicity_comments = []
    for comment in comments:
        score = get_toxicity(comment)

        if 0.4 <= score < 0.6:
            mid_toxicity_comments.append({
                'comment': comment,
                'toxicity_score': score,
                'flagged_status': True
            })
        elif score >= 0.6:
            high_toxicity_comments.append({
                'comment': comment,
                'toxicity_score': score,
                'flagged_status': True
            })

    
    return {
        "mid_toxicity_comments": mid_toxicity_comments,
        "high_toxicity_comments": high_toxicity_comments
    }



    
