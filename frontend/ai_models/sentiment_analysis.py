import nltk
# import kagglehub
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re 
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "ai_models", "datasets", "sentimentdataset.csv")



nltk.download('stopwords')


data = fetch_20newsgroups(subset='train', categories=['rec.autos', 'rec.sport.baseball'])
texts = data.data
labels = data.target

df = pd.read_csv(DATA_PATH)
texts = df["review"].tolist()
labels = df["sentiment"].map({'positive': 1, 'negative': 0}).tolist()

stopwords = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

cleaned_texts = [clean_text(text) for text in texts]

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    max_df= 0.9,
    min_df= 5
    )
X_features= tfidf_vectorizer.fit_transform(cleaned_texts)
X_train, X_test, y_train, y_test = train_test_split(
    X_features, labels, test_size=0.2, random_state=42, stratify=labels
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

def predict_sentiment(text:str)-> str:
    cleaned = clean_text(text)
    features = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return 'positive' if prediction == 1 else 'negative'