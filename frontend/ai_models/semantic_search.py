import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os

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
        embedded = self.embedding(x)              # [batch, seq_len, embed_dim]
        _, (hn, _) = self.lstm(embedded)         # hn: [num_layers*2, batch, hidden_dim]
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate last hidden states from both directions
        out = self.fc(hn_cat)                     # [batch, 64]
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
    """Convert a single text to tensor of indices."""
    tokens = text.lower().split()
    idxs = [word_to_idx.get(t, word_to_idx['<unk>']) for t in tokens]
    if len(idxs) > max_len:
        idxs = idxs[:max_len]
    else:
        idxs += [word_to_idx['<pad>']] * (max_len - len(idxs))
    return torch.tensor([idxs], device=device)  # [1, max_len]

def load_model(checkpoint_path, embedding_matrix, device):
    model = SiameseLSTM(embedding_matrix).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# -----------------------------
# Batch Encoding for Candidates
# -----------------------------
def batch_encode(texts, model, word_to_idx, max_len=100, device='cpu', batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_tensors = torch.cat([text_to_tensor(t, word_to_idx, max_len, device) for t in batch_texts])
        with torch.no_grad():
            batch_embs = model.encode(batch_tensors)  # [batch, 64]
        embeddings.append(batch_embs)
    return torch.cat(embeddings, dim=0)  # [num_texts, 64]

# -----------------------------
# Semantic Search
# -----------------------------
def semantic_search(query, model, word_to_idx, candidate_texts, candidate_embs, device, top_k=5, max_len=100):
    """Perform semantic search for a query."""
    with torch.no_grad():
        query_tensor = text_to_tensor(query, word_to_idx, max_len=max_len, device=device)
        query_emb = model.encode(query_tensor)                  # [1, 64]

        # Normalize embeddings for cosine similarity
        query_emb_norm = nn.functional.normalize(query_emb, dim=1)
        candidate_embs_norm = nn.functional.normalize(candidate_embs, dim=1)

        sims = torch.mm(query_emb_norm, candidate_embs_norm.T).squeeze()  # [num_candidates]
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

    # Load trained Siamese LSTM model
    checkpoint_path = './trained_models/siamese_lstm_final.pth'
    model = load_model(checkpoint_path, embedding_matrix, device)

    # Load candidate texts
    candidates_df = pd.read_csv('./datasets/comments_to_search.txt', sep='\t')
    candidate_texts = candidates_df['comment_text'].tolist()

    # Encode all candidate texts in batches
    candidate_embs = batch_encode(candidate_texts, model, vocab, max_len=max_len, device=device)

    # Example user query
    user_query = "How to improve thread quality and reduce toxicity?"

    # Perform semantic search
    results = semantic_search(user_query, model, vocab, candidate_texts, candidate_embs, device, top_k=5, max_len=max_len)

    # Display results
    print("Top semantic search results:")
    for text, score in results:
        print(f"Score: {score:.4f} - Text: {text}")
