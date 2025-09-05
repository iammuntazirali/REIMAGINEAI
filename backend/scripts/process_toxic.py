import os
import json
from fastapi import FastAPI
from ai_models.toxicity import check_thread_toxicity, ThreadRequest

app = FastAPI()

# Define the folder path (parent directory's 'data' folder)
data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Ensure the 'data' folder exists
os.makedirs(data_folder, exist_ok=True)

@app.post("/process_toxicity/")
async def process_toxicity(request_data: ThreadRequest):
    result = check_thread_toxicity(request_data)

    # Paths for saving JSON files
    mid_toxicity_path = os.path.join(data_folder, "mid_toxicity.json")
    high_toxicity_path = os.path.join(data_folder, "toxicity.json")

    # Save mid toxicity comments (0.4 to 0.6)
    with open(mid_toxicity_path, 'w', encoding='utf-8') as f:
        json.dump(result.get("mid_toxicity_comments", []), f, ensure_ascii=False, indent=4)

    # Save high toxicity comments (>= 0.6)
    with open(high_toxicity_path, 'w', encoding='utf-8') as f:
        json.dump(result.get("high_toxicity_comments", []), f, ensure_ascii=False, indent=4)

    return result
