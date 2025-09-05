import requests
import json

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


def flag_toxic_comment(thread, threshold):
    flagged_comments = []
    for comment in thread:

        score = get_toxicity(comment)
        flagged = score >= threshold
        
        flagged_comments.append({
            'comment': comment,
            'toxicity_score': score,
            'flagged_status': flagged
            })
        
    return flagged_comments

    


if __name__ == "__main__":
    thread = ["Hey nice to meet you", "you are not a good person", "you are such a baddie", "there are people worse than you", "you can never beat me ever", "I hate you", "you should just leave this earth", "just go and die, just kidding", "fuck you a million times", "Jiminy cricket! Well gosh durned it! Oh damn it all!","Rohan is a nigga", "I love you"]

    results = flag_toxic_comment(thread, 0.6)
    for result in results:
        print(f"Comment : {result["comment"]}\nToxicity Score = {result["toxicity_score"]} \nFlagged : {result["flagged_status"]}")