import os
import re
import math
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import pearsonr

from gensim.models import Word2Vec
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
@dataclass
class Config:
    train_file: str = "./ai_models/datasets/msr_paraphrase_train.txt"
    test_file: str  = "./ai_models/datasets/msr_paraphrase_test.txt"
    artifacts_dir: str = "./ai_models/trained_models"
    checkpoints_dir: str = "./ai_models/trained_models/checkpoints"
    max_len: int = 100
    embedding_dim: int = 100
    hidden_dim: int = 512
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 12
    weight_decay: float = 0.0
    patience: int = 3  
    margin: float = 0.6
    val_size: float = 0.1
    seed: int = 42
    num_workers: int = 0  

CFG = Config()
os.makedirs(CFG.artifacts_dir, exist_ok=True)
os.makedirs(CFG.checkpoints_dir, exist_ok=True)

# -------------------------------
# Utility Functions
# -------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def simple_tokenize(text: str):
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]

def load_and_prepare_data(train_path, test_path):
    df_train = pd.read_csv(train_path, sep='\t', on_bad_lines='skip')
    df_train = df_train.rename(columns={'Quality': 'label', '#1 String': 'comment1', '#2 String': 'comment2'})
    df_train = df_train[['comment1', 'comment2', 'label']].dropna()

    df_test = pd.read_csv(test_path, sep='\t', on_bad_lines='skip')
    df_test = df_test.rename(columns={'Quality': 'label', '#1 String': 'comment1', '#2 String': 'comment2'})
    df_test = df_test[['comment1', 'comment2', 'label']].dropna()
    return df_train, df_test

def build_vocab(text_series_iterable, min_freq: int = 1):
    vocab = {'<pad>': 0, '<unk>': 1}
    freq = {}
    for text in text_series_iterable:
        for tok in simple_tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1
    idx = 2
    for tok, c in freq.items():
        if c >= min_freq:
            vocab[tok] = idx
            idx += 1
    return vocab

# -------------------------------
# Word2Vec Embeddings
# -------------------------------
def train_word2vec(sentences, embedding_dim=100, seed=42):
    tokenized_sentences = [simple_tokenize(s) for s in sentences]
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=4,
        seed=seed,
        sg=1
    )
    return model

def build_embedding_matrix_from_w2v(vocab, w2v_model, embedding_dim=100):
    emb_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim)).astype('float32')
    emb_matrix[vocab['<pad>']] = np.zeros(embedding_dim, dtype='float32')
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            emb_matrix[idx] = w2v_model.wv[word]
    return emb_matrix

# -------------------------------
# Dataset
# -------------------------------
class CommentPairsDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len=100):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def text_to_indices(self, text):
        toks = simple_tokenize(text)
        idxs = [self.vocab.get(t, self.vocab['<unk>']) for t in toks][:self.max_len]
        length = len(idxs)
        if length < self.max_len:
            idxs += [self.vocab['<pad>']] * (self.max_len - length)
        return np.array(idxs, dtype=np.int64), length

    def __getitem__(self, i):
        row = self.df.iloc[i]
        c1_idx, len1 = self.text_to_indices(row['comment1'])
        c2_idx, len2 = self.text_to_indices(row['comment2'])
        label = float(row['label'])
        return (
            torch.tensor(c1_idx),
            torch.tensor(len1, dtype=torch.int64),
            torch.tensor(c2_idx),
            torch.tensor(len2, dtype=torch.int64),
            torch.tensor(label, dtype=torch.float32)
        )

def collate_fn(batch):
    c1, l1, c2, l2, y = zip(*batch)
    return (
        torch.stack(c1, dim=0),
        torch.stack(l1, dim=0),
        torch.stack(c2, dim=0),
        torch.stack(l2, dim=0),
        torch.stack(y, dim=0)
    )

# -------------------------------
# Model
# -------------------------------
class SiameseLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.2):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        hn_cat = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        out = torch.relu(self.proj(hn_cat))
        out = self.dropout(out)
        return out

    def forward(self, c1, l1, c2, l2):
        e1 = self.encode(c1, l1)
        e2 = self.encode(c2, l2)
        sim = nn.functional.cosine_similarity(e1, e2)
        return sim

# -------------------------------
# Loss
# -------------------------------
def contrastive_loss(sim, label, margin=0.5):
    loss_similar = label * (1 - sim) ** 2
    loss_dissimilar = (1 - label) * torch.clamp(sim - margin, min=0) ** 2
    return torch.mean(loss_similar + loss_dissimilar)

# -------------------------------
# Training & Evaluation
# -------------------------------
def train_one_epoch(model, loader, optimizer, device, margin):
    model.train()
    total = 0.0
    for c1, l1, c2, l2, y in loader:
        c1, l1, c2, l2, y = c1.to(device), l1.to(device), c2.to(device), l2.to(device), y.to(device)
        optimizer.zero_grad()
        sim = model(c1, l1, c2, l2)
        loss = contrastive_loss(sim, y, margin=margin)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, device, margin):
    model.eval()
    losses, sims, labels = [], [], []
    for c1, l1, c2, l2, y in loader:
        c1, l1, c2, l2, y = c1.to(device), l1.to(device), c2.to(device), l2.to(device), y.to(device)
        sim = model(c1, l1, c2, l2)
        loss = contrastive_loss(sim, y, margin=margin)
        losses.append(loss.item())
        sims.append(sim.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    sims = np.concatenate(sims) if sims else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    return (np.mean(losses) if losses else math.nan), sims, labels

def pick_best_threshold(sims_val, labels_val):
    probs = (sims_val + 1.0) / 2.0
    best_thr, best_f1, best_acc = 0.5, -1.0, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels_val, preds)
        acc = accuracy_score(labels_val, preds)
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and acc > best_acc):
            best_f1, best_acc, best_thr = f1, acc, thr
    return best_thr, best_f1, best_acc, probs

def compute_metrics(sims, labels, thr):
    probs = (sims + 1.0) / 2.0
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')
    try:
        pr, _ = pearsonr(sims, labels)
    except Exception:
        pr = float('nan')
    return {"accuracy": acc, "f1": f1, "roc_auc": auc, "pearson": pr}

# -------------------------------
# Main
# -------------------------------
def main():
    set_seed(CFG.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load data
    df_train_full, df_test = load_and_prepare_data(CFG.train_file, CFG.test_file)
    train_df, val_df = train_test_split(
        df_train_full, test_size=CFG.val_size, random_state=CFG.seed, stratify=df_train_full['label']
    )

    # Build vocab
    vocab = build_vocab(pd.concat([train_df['comment1'], train_df['comment2']]))
    print(f"Vocab size: {len(vocab)}")

    # Train Word2Vec
    all_sentences = pd.concat([train_df['comment1'], train_df['comment2']])
    w2v_model = train_word2vec(all_sentences, embedding_dim=CFG.embedding_dim)

    # Build embedding matrix from Word2Vec
    embedding_matrix = build_embedding_matrix_from_w2v(vocab, w2v_model, embedding_dim=CFG.embedding_dim)

    # Save vocab and embeddings
    with open(os.path.join(CFG.artifacts_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    np.save(os.path.join(CFG.artifacts_dir, "embedding_matrix.npy"), embedding_matrix)

    # Datasets & loaders
    train_ds = CommentPairsDataset(train_df, vocab, max_len=CFG.max_len)
    val_ds   = CommentPairsDataset(val_df,   vocab, max_len=CFG.max_len)
    test_ds  = CommentPairsDataset(df_test,  vocab, max_len=CFG.max_len)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model & optimizer
    model = SiameseLSTM(embedding_matrix, hidden_dim=CFG.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_val_loss = float('inf')
    best_path = os.path.join(CFG.artifacts_dir, "siamese_lstm_best.pth")
    patience_left = CFG.patience

    # Training loop
    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, CFG.margin)
        val_loss, val_sims, val_labels = eval_epoch(model, val_loader, device, CFG.margin)
        scheduler.step(val_loss)

        thr, f1, acc, _ = pick_best_threshold(val_sims, val_labels)
        metrics = compute_metrics(val_sims, val_labels, thr)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            patience_left = CFG.patience
            print("  --> Best model saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered")
                break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(best_path))
    test_loss, test_sims, test_labels = eval_epoch(model, test_loader, device, CFG.margin)
    thr, f1, acc, _ = pick_best_threshold(test_sims, test_labels)
    metrics = compute_metrics(test_sims, test_labels, thr)
    print(f"Test metrics: Loss={test_loss:.4f}, F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, Pearson={metrics['pearson']:.4f}")

if __name__ == "__main__":
    main()


