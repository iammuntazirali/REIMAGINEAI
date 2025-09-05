import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle

# -----------------------------
# Siamese LSTM Definition
# -----------------------------
class SiameseLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128):
        super(SiameseLSTM, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix).float(), freeze=False
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64)

    def encode(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1)  # concatenate last hidden states
        out = self.fc(hn_cat)
        return torch.relu(out)


# -----------------------------
# Utilities
# -----------------------------
def load_vocab(path='./vocab.pkl'):
    """Load vocabulary dictionary saved during training."""
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    """Load GloVe embeddings and map to vocab."""
    embeddings = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim))
    embeddings[vocab['<pad>']] = np.zeros(embedding_dim)  # zero vector for padding

    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            if word in vocab:
                embeddings[vocab[word]] = vector
    return embeddings

def text_to_tensor(text, word_to_idx, max_len=100, device='cpu'):
    tokens = text.lower().split()
    idxs = [word_to_idx.get(t, word_to_idx['<unk>']) for t in tokens]
    if len(idxs) > max_len:
        idxs = idxs[:max_len]
    else:
        idxs += [word_to_idx['<pad>']] * (max_len - len(idxs))
    return torch.tensor([idxs], device=device)  # batch of 1

def load_model(checkpoint_path, embedding_matrix, device):
    model = SiameseLSTM(embedding_matrix).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# -----------------------------
# Semantic Search
# -----------------------------
def semantic_search(query, model, word_to_idx, candidate_texts, candidate_embs, device, top_k=5, max_len=100):
    """Perform semantic search for a query."""
    with torch.no_grad():
        query_tensor = text_to_tensor(query, word_to_idx, max_len=max_len, device=device)
        query_emb = model.encode(query_tensor)

        # Vectorized cosine similarity
        sims = nn.functional.cosine_similarity(query_emb, candidate_embs, dim=1)
        top_indices = sims.topk(top_k).indices.tolist()

    return [(candidate_texts[i], sims[i].item()) for i in top_indices]

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = 100  # sequence length

    # Load vocab and embeddings
    vocab = load_vocab('./vocab.pkl')
   
    embedding_matrix = load_glove_embeddings('./glove.6B.100d.txt', vocab, embedding_dim=100)
  

    checkpoint_path = './trained_models/siamese_lstm_final.pth'
    model = load_model(checkpoint_path, embedding_matrix, device)

    
    candidates_df = pd.read_csv('./datasets/comments_to_search.txt', sep='\t')
    candidate_texts = candidates_df['comment_text'].tolist()

    
    candidate_embs = torch.cat([
        model.encode(text_to_tensor(text, vocab, max_len=max_len, device=device))
        for text in candidate_texts
    ], dim=0)

    
    user_query = "How to improve thread quality and reduce toxicity?"

   
    results = semantic_search(user_query, model, vocab, candidate_texts, candidate_embs, device, top_k=5, max_len=max_len)


    print("Top semantic search results:")
    for text, score in results:
        print(f"Score: {score:.4f} - Text: {text}")

