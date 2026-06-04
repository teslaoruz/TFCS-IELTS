from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer, set_seed

from src.rag.config import resolve_torch_device


class DistilBertRegressor(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.regressor(pooled).squeeze(-1)


class EssayDataset(Dataset):
    def __init__(self, texts: list[str], scores: list[float] | None, tokenizer, max_length: int = 256):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.scores is not None:
            item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item

def train_distilbert(
    df_train: pd.DataFrame,
    band_column: str = "overall",
    text_column: str = "essay",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[DistilBertRegressor, DistilBertTokenizer]:
    model_name = str(model_name)
    max_length = int(max_length)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    num_epochs = int(num_epochs)
    hidden_dim = int(hidden_dim)
    dropout = float(dropout)

    device = resolve_torch_device(device)
    use_cuda = device.startswith("cuda")
    set_seed(seed)
    if use_cuda:
        torch.backends.cudnn.benchmark = False

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertRegressor(
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    train_texts = df_train[text_column].fillna("").astype(str).tolist()
    train_scores = pd.to_numeric(df_train[band_column], errors="coerce").astype(float).tolist()
    train_dataset = EssayDataset(train_texts, train_scores, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
            attention_mask = batch["attention_mask"].to(device, non_blocking=use_cuda)
            scores = batch["score"].to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_distilbert(
    model: DistilBertRegressor,
    tokenizer: DistilBertTokenizer,
    df_test: pd.DataFrame,
    text_column: str = "essay",
    max_length: int = 256,
    batch_size: int = 16,
    device: str = "cpu",
    clip_range: tuple[float, float] = (0.0, 9.0),
) -> np.ndarray:
    device = resolve_torch_device(device)
    use_cuda = device.startswith("cuda")
    max_length = int(max_length)
    batch_size = int(batch_size)
    test_texts = df_test[text_column].fillna("").astype(str).tolist()
    test_dataset = EssayDataset(test_texts, None, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
    )

    model.eval()
    all_preds = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
        attention_mask = batch["attention_mask"].to(device, non_blocking=use_cuda)
        preds = model(input_ids, attention_mask)
        all_preds.extend(preds.cpu().numpy())

    preds = np.array(all_preds)
    preds = np.clip(preds, clip_range[0], clip_range[1])
    return np.round(preds * 2) / 2
