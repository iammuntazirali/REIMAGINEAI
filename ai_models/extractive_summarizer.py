from summarizer import Summarizer
 # for extractive
from transformers import pipeline  # for abstractive

# 1) Extractive summarizer instance
extractive_model = Summarizer()

# 2) Abstractive summarizer pipeline
abstractive_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    device=0  # GPU if available
)

def summarize(text: str, method: str = "abstractive", max_length: int = 150) -> str:
    if method == "extractive":
        return extractive_model(text, ratio=0.2)
    # abstractive
    summary = abstractive_pipeline(
        text,
        max_length=max_length,
        min_length=30,
        do_sample=False
    )
    return summary[0]["summary_text"]