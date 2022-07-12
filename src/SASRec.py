from typing import Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=(1,))
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=(1,))
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    def __init__(
            self,
            item_num: int,
            hidden_units: int,
            max_len: int,
            dropout_rate: float,
            num_blocks: int,
            num_heads: int
    ):
        super().__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num + 1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layer_norms = nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layer_norms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layer_norm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layer_norms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            new_attn_layer = nn.MultiheadAttention(
                hidden_units, num_heads, dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            self.forward_layer_norms.append(nn.LayerNorm(hidden_units, eps=1e-8))

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layer_norms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layer_norms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layer_norm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(
            self,
            log_seqs: torch.Tensor,
            pos_seqs: torch.Tensor,
            neg_seqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def score(self, seq):
        """
        Takes 1d sequence as input and returns prediction scores.
        """
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


def data_to_sequences(data: pd.DataFrame) -> pd.Series:
    sequences = (
        data.sort_values(['userid', 'timestamp'])
            .groupby('userid', sort=False)['movieid'].apply(list)
    )
    return sequences


def sequential_batch_sampler(
        user_train: pd.Series,
        user_num: int,
        item_num: int,
        batch_size: int,
        max_len: int,
        seed: int,
        pad_token: Optional[int] = None
):
    if pad_token is None:
        pad_token = item_num

    def random_neq(l: int, r: int, set_items: Set):
        t = random_state.randint(l, r)
        while t in set_items:
            t = random_state.randint(l, r)
        return t

    def sample() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        user = random_state.randint(user_num)
        while len(user_train.get(user, [])) <= 1:
            user = random_state.randint(user_num)
        user_items = user_train[user]
        seq = np.full(max_len, pad_token, dtype=np.int32)
        pos = np.full(max_len, pad_token, dtype=np.int32)
        neg = np.full(max_len, pad_token, dtype=np.int32)
        nxt = user_items[-1]
        idx = max_len - 1
        ts = set(user_items)
        for i in reversed(user_items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(0, item_num, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return user, seq, pos, neg

    random_state = np.random.RandomState(seed)
    while True:
        yield zip(*(sample() for _ in range(batch_size)))


def train_epoch(model, num_batch, sampler, optimizer, criterion, device) -> float:
    model.train()
    pad_token = model.pad_token
    losses = []
    for _ in tqdm(range(num_batch)):
        _, *seq_data = next(sampler)
        # convert batch data into torch tensors
        seq, pos, neg = (torch.LongTensor(np.array(x)).to(device) for x in seq_data)
        pos_logits, neg_logits = model(seq, pos, neg)
        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)
        optimizer.zero_grad()
        indices = np.where(pos != pad_token)
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        # if l2_emb != 0:
        #     for param in model.item_emb.parameters():
        #         loss += l2_emb * torch.norm(param)**2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def save_checkpoint(model, epoch, optimizer, cfg):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, cfg.SASRec.model_save_path)


def train_sasrec(data, cfg):
    logger.info("start learning SASRec")

    device = cfg.SASRec.device
    user_num = max(data['userid'].unique())
    item_num = max(data['movieid'].unique())

    model = SASRec(item_num, **cfg.SASRec.model_params).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        **cfg.SASRec.optimizer_params
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    train_sequences = data_to_sequences(data)

    sampler = sequential_batch_sampler(
        user_train=train_sequences,
        user_num=user_num,
        item_num=item_num,
        pad_token=model.pad_token,
        **cfg.SASRec.batch_sampler_params
    )

    losses = {}
    num_batch = len(train_sequences) // cfg.SASRec.batch_sampler_params.batch_size
    for epoch in tqdm(range(cfg.SASRec.num_epoch)):
        cur_loss = train_epoch(
            model, num_batch, sampler, optimizer, criterion, device)
        save_checkpoint(model, epoch, optimizer, cfg)
        losses[epoch] = cur_loss
        logger.info(f"epoch: {epoch}, loss: {cur_loss}")
    return model, losses


def evaluate_sasrec(model, data, device):
    model.eval()
    test_sequences = data_to_sequences(data)
    scores = []
    with torch.no_grad():
        for _, seq in test_sequences.items():
            predictions = model.score(torch.tensor(seq).to(device))
            scores.append(predictions.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)
