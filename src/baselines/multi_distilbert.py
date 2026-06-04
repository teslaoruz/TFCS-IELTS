"""
Multi-output DistilBERT regressor predicting 4 criterion subscores + overall.
This is the strongest baseline and the scoring backbone of the proposed pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer


class MultiDistilBertRegressor(nn.Module):
    """Predicts 5 outputs: task_response, coherence, lexical, grammar, overall."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.15, n_outputs: int = 5):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.regressor(pooled)


class MultiScoreDataset(Dataset):
    def __init__(self, texts, scores_array, tokenizer, max_length=256):
        self.texts = texts
        self.scores = torch.tensor(scores_array, dtype=torch.float) if scores_array is not None else None
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
            item["scores"] = self.scores[idx]
        return item


def train_multi_distilbert(
    df_train: pd.DataFrame,
    text_column: str = "essay",
    score_columns: list[str] | None = None,
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 4,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    device: str = "cpu",
    verbose: bool = True,
    loss_weights: list[float] | None = None,
) -> tuple[MultiDistilBertRegressor, DistilBertTokenizer, list[str]]:
    if score_columns is None:
        score_columns = ["task_response", "coherence", "lexical", "grammar", "overall"]

    avail_cols = [c for c in score_columns if c in df_train.columns]
    missing = [c for c in score_columns if c not in df_train.columns]
    if missing and verbose:
        print(f"  Missing score columns (will omit): {missing}")

    df_clean = df_train[avail_cols].dropna()
    train_texts = df_train.loc[df_clean.index, text_column].values.tolist()
    train_scores = df_clean.values.astype(np.float32)

    if verbose:
        print(f"  Training samples with all scores: {len(train_texts)}")
        for i, col in enumerate(avail_cols):
            print(f"    {col}: mean={train_scores[:, i].mean():.2f}, std={train_scores[:, i].std():.2f}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = MultiDistilBertRegressor(
        hidden_dim=hidden_dim, dropout=dropout, n_outputs=len(avail_cols)
    ).to(device)

    dataset = MultiScoreDataset(train_texts, train_scores, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if loss_weights is not None and len(loss_weights) == len(avail_cols):
        weight_tensor = torch.tensor(loss_weights, dtype=torch.float32, device=device)
    else:
        weight_tensor = None
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["scores"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = F.mse_loss(preds, scores, reduction='none')
            if weight_tensor is not None:
                loss = (loss * weight_tensor).mean()
            else:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if verbose:
            print(f"  Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.6f}")

    model.eval()
    return model, tokenizer, avail_cols


@torch.no_grad()
def predict_multi_distilbert(
    model: MultiDistilBertRegressor,
    tokenizer: DistilBertTokenizer,
    df_test: pd.DataFrame,
    text_column: str = "essay",
    max_length: int = 256,
    batch_size: int = 16,
    device: str = "cpu",
    clip_range: tuple[float, float] = (0.0, 9.0),
) -> dict[str, np.ndarray]:
    test_texts = df_test[text_column].values.tolist()
    dataset = MultiScoreDataset(test_texts, None, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        preds = model(input_ids, attention_mask)
        all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_preds = np.clip(all_preds, clip_range[0], clip_range[1])
    all_preds = np.round(all_preds * 2) / 2

    result = {}
    for i in range(all_preds.shape[1]):
        result[f"pred_{i}"] = all_preds[:, i]
    return result
