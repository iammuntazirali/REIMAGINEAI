import os
import re
import json
import math
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import pearsonr


@dataclass
class Config:
    train_file: str = "./ai_models/datasets/msr_paraphrase_train.txt"
    test_file: str  = "./ai_models/datasets/msr_paraphrase_test.txt"
    glove_path: str = "./Glove_vector_embedding.txt"     
    artifacts_dir: str = "./ai_models/trained_models"
    checkpoints_dir: str = "./ai_models/trained_models/checkpoints"
    max_len: int = 100
    embedding_dim: int = 100
    hidden_dim: int = 128
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 12
    weight_decay: float = 0.0
    patience: int = 3  
    margin: float = 0.5  
    val_size: float = 0.1
    seed: int = 42
    num_workers: int = 0  

CFG = Config()

os.makedirs(CFG.artifacts_dir, exist_ok=True)
os.makedirs(CFG.checkpoints_dir, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simple_tokenize(text: str):
    
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


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

def load_glove_embeddings(glove_path, vocab, embedding_dim):
    
    if not os.path.isfile(glove_path):
        raise FileNotFoundError(f"GloVe file not found at {glove_path}")

    vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            word = parts[0]
            vec = parts[1:]
            if len(vec) != embedding_dim:
           
                raise ValueError(
                    f"GloVe vector dimension {len(vec)} does not match embedding_dim={embedding_dim}. "
                    f"Use the correct GloVe file (e.g., glove.6B.{embedding_dim}d.txt)."
                )
            vectors[word] = np.asarray(vec, dtype='float32')

    print(f"Loaded {len(vectors)} GloVe word vectors.")

   
    emb = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim)).astype('float32')
    emb[vocab['<pad>']] = np.zeros(embedding_dim, dtype='float32')

   
    if len(vectors) > 0:
        mean_vec = np.mean(np.stack(list(vectors.values()), axis=0), axis=0)
    else:
        mean_vec = np.zeros(embedding_dim, dtype='float32')
    emb[vocab['<unk>']] = mean_vec

    hit = 0
    for w, i in vocab.items():
        if w in ('<pad>', '<unk>'):
            continue
        if w in vectors:
            emb[i] = vectors[w]
            hit += 1
    cov = hit / max(1, (len(vocab) - 2))
    print(f"GloVe coverage on vocab: {hit}/{len(vocab)-2} = {cov:.2%}")
    return emb


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


class SiameseLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 64)

    def encode(self, x, lengths):
      
        embedded = self.embedding(x)  
      
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)              # hn: [2, B, H]
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1) # [B, 2H]
        out = torch.relu(self.proj(hn_cat))         # [B, D]
        return out

    def forward(self, c1, l1, c2, l2):
        e1 = self.encode(c1, l1)
        e2 = self.encode(c2, l2)
        sim = nn.functional.cosine_similarity(e1, e2)  # [B]
        return sim


def contrastive_loss(sim, label, margin=0.5):
 
    loss_similar = label * (1 - sim) ** 2
    loss_dissimilar = (1 - label) * torch.clamp(sim - margin, min=0) ** 2
    return torch.mean(loss_similar + loss_dissimilar)


def train_one_epoch(model, loader, optimiz, device, margin):
    model.train()
    total = 0.0
    for c1, l1, c2, l2, y in loader:
        c1, l1, c2, l2, y = c1.to(device), l1.to(device), c2.to(device), l2.to(device), y.to(device)
        optimiz.zero_grad()
        sim = model(c1, l1, c2, l2)
        loss = contrastive_loss(sim, y, margin=margin)
        loss.backward()
        optimiz.step()
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


def main():
    set_seed(CFG.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 1) Load data
    df_train_full, df_test = load_and_prepare_data(CFG.train_file, CFG.test_file)

   
    train_df, val_df = train_test_split(
        df_train_full, test_size=CFG.val_size, random_state=CFG.seed, stratify=df_train_full['label']
    )

    
    vocab = build_vocab(pd.concat([train_df['comment1'], train_df['comment2']]))
    print(f"Vocab size: {len(vocab)}")

 
    embedding_matrix = load_glove_embeddings(CFG.glove_path, vocab, CFG.embedding_dim)
    
    with open(os.path.join(CFG.artifacts_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    np.save(os.path.join(CFG.artifacts_dir, "embedding_matrix.npy"), embedding_matrix)

 
    train_ds = CommentPairsDataset(train_df, vocab, max_len=CFG.max_len)
    val_ds   = CommentPairsDataset(val_df,   vocab, max_len=CFG.max_len)
    test_ds  = CommentPairsDataset(df_test,  vocab, max_len=CFG.max_len)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)

    
    model = SiameseLSTM(embedding_matrix, hidden_dim=CFG.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

  
    best_val_loss = float('inf')
    best_path = os.path.join(CFG.artifacts_dir, "siamese_lstm_best.pth")
    patience_left = CFG.patience

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, CFG.margin)
        val_loss, sims_val, labels_val = eval_epoch(model, val_loader, device, CFG.margin)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f}")
        # Save checkpoint each epoch
        ckpt_path = os.path.join(CFG.checkpoints_dir, f"siamese_lstm_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

     
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            patience_left = CFG.patience
            print("  ↳ Saved new best model.")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break

 
    model.load_state_dict(torch.load(best_path, map_location=device))


    _, sims_val, labels_val = eval_epoch(model, val_loader, device, CFG.margin)
    best_thr, best_f1, best_acc, val_probs = pick_best_threshold(sims_val, labels_val)
    print(f"Best threshold (on val): {best_thr:.3f} | Val F1: {best_f1:.4f} | Val Acc: {best_acc:.4f}")
    with open(os.path.join(CFG.artifacts_dir, "threshold.txt"), "w") as f:
        f.write(str(best_thr))

    _, sims_test, labels_test = eval_epoch(model, test_loader, device, CFG.margin)
    metrics = compute_metrics(sims_test, labels_test, best_thr)
    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

   
    final_model_path = os.path.join(CFG.artifacts_dir, "siamese_lstm_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved artifacts to: {CFG.artifacts_dir}")

if __name__ == "__main__":
    main()
