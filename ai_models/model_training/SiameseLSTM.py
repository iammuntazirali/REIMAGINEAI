import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import os


class CommentPairsDataset(Dataset):
    def __init__(self, dataframe, word_to_idx, max_len=100):
        self.data = dataframe
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_to_indices(self, text):
        tokens = text.lower().split()
        idxs = [self.word_to_idx.get(t, self.word_to_idx['<unk>']) for t in tokens]
        if len(idxs) > self.max_len:
            idxs = idxs[:self.max_len]
        else:
            idxs += [self.word_to_idx['<pad>']] * (self.max_len - len(idxs))
        return np.array(idxs)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        c1 = self.text_to_indices(row['comment1'])
        c2 = self.text_to_indices(row['comment2'])
        label = float(row['label'])
        return torch.tensor(c1), torch.tensor(c2), torch.tensor(label)


class SiameseLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128):
        super(SiameseLSTM, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64)

    def encode(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1)  # last hidden states of both directions
        out = self.fc(hn_cat)
        out = torch.relu(out)
        return out

    def forward(self, c1, c2):
        e1 = self.encode(c1)
        e2 = self.encode(c2)
        return nn.functional.cosine_similarity(e1, e2)


def contrastive_loss(sim, label, margin=0.5):
    loss_similar = label * (1 - sim) ** 2
    loss_dissimilar = (1 - label) * torch.clamp(sim - margin, min=0) ** 2
    return torch.mean(loss_similar + loss_dissimilar)


def load_and_prepare_data(train_path, test_path):
    df_train = pd.read_csv(train_path, sep='\t', on_bad_lines='skip')
    df_train = df_train.rename(columns={
        'Quality': 'label',
        '#1 String': 'comment1',
        '#2 String': 'comment2'
    })[['comment1', 'comment2', 'label']]

    df_test = pd.read_csv(test_path, sep='\t', on_bad_lines='skip')
    df_test = df_test.rename(columns={
        'Quality': 'label',
        '#1 String': 'comment1',
        '#2 String': 'comment2'
    })[['comment1', 'comment2', 'label']]

    return df_train, df_test


def build_vocab_and_embeddings(df_train, embedding_dim=100):
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for text in pd.concat([df_train['comment1'], df_train['comment2']]):
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim))
    embedding_matrix[vocab['<pad>']] = np.zeros(embedding_dim)  # zero vector for padding
    return vocab, embedding_matrix


def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for c1, c2, label in dataloader:
        c1, c2, label = c1.to(device), c2.to(device), label.to(device)
        optimizer.zero_grad()
        sim = model(c1, c2)
        loss = contrastive_loss(sim, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    sims, labels = [], []
    with torch.no_grad():
        for c1, c2, label in dataloader:
            c1, c2 = c1.to(device), c2.to(device)
            sim = model(c1, c2)
            sims.extend(sim.cpu().numpy())
            labels.extend(label.numpy())
    return sims, labels


if __name__ == '__main__':
    train_file = './ai_models/datasets/msr_paraphrase_train.txt'
    test_file = './ai_models/datasets/msr_paraphrase_test.txt'

    df_train, df_test = load_and_prepare_data(train_file, test_file)
    vocab, embedding_matrix = build_vocab_and_embeddings(df_train, embedding_dim=100)

    train_dataset = CommentPairsDataset(df_train, vocab)
    test_dataset = CommentPairsDataset(df_test, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseLSTM(embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_dir = './ai_models/trained_models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f'siamese_lstm_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    sims, labels = evaluate_model(model, test_loader, device)
    for i in range(10):
        print(f"Similarity: {sims[i]:.4f}, Label: {labels[i]}")

    final_model_path = './ai_models/trained_models/siamese_lstm_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

