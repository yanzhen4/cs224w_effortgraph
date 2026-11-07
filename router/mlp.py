"""Non-graph MLP router (the proposal's comparison for the GNN): same query/task features, no message passing.

The query and task-group embeddings each pass through a two-layer encoder (Linear-ReLU-Linear, hidden 32, standing
in for the GNN's two GAT layers); models and efforts get their own learned 32-d embeddings; the scorer is the
proposal's MLP over [h_q | h_t | h_m | h_e] (eq. 2). Training follows the proposal: binary cross-entropy with the
highest-reward configuration as the positive (eq. 3), Adam at 1e-3, batch of 32 questions (each with its 9
configurations), 1,000 epochs, final checkpoint, no validation set. Every (alpha, beta) setting x fold x seed is
one training job; the test fold's picks go to runs/router/mlp/picks.jsonl.

    python router/mlp.py            # 3 settings x 5 folds x 5 seeds
"""
import argparse
import json
import multiprocessing
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import data

HIDDEN = 32
EPOCHS = 1000
BATCH = 32
LR = 1e-3
MODEL_INDEX = torch.tensor([list(data.MODELS).index(m) for m, _ in data.CONFIGS])
EFFORT_INDEX = torch.tensor([data.EFFORTS.index(e) for _, e in data.CONFIGS])


class MLPRouter(nn.Module):
    def __init__(self, d_text, hidden=HIDDEN):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(d_text, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.task = nn.Sequential(nn.Linear(d_text, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.model = nn.Embedding(len(data.MODELS), hidden)
        self.effort = nn.Embedding(len(data.EFFORTS), hidden)
        self.head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, xq, xt):
        """(batch, d) query and task embeddings -> (batch, configs) scores."""
        b, k = xq.shape[0], len(data.CONFIGS)
        hq, ht = self.query(xq), self.task(xt)
        hm, he = self.model(MODEL_INDEX), self.effort(EFFORT_INDEX)
        z = torch.cat([hq[:, None].expand(b, k, -1), ht[:, None].expand(b, k, -1),
                       hm[None].expand(b, k, -1), he[None].expand(b, k, -1)], dim=-1)
        return self.head(z).squeeze(-1)


def features(table):
    emb = np.load(data.ROOT / 'data/embeddings/all-MiniLM-L6-v2.npz')
    assert list(emb['ids']) == table.ids
    task = emb['task'][[list(emb['groups']).index(g) for g in table.group]]
    return torch.from_numpy(emb['query']), torch.from_numpy(task)


TABLE = XQ = XT = None


def init():
    global TABLE, XQ, XT
    torch.set_num_threads(1)
    TABLE = data.load()
    XQ, XT = features(TABLE)


def train_job(job):
    (alpha, beta), fold, seed = job
    t = TABLE
    train, test = t.fold != fold, t.fold == fold
    R = data.Normalizer(t.P[train], t.C[train]).reward(t.P[train], t.C[train], alpha, beta)
    labels = data.best(R, t.C[train])
    Y = F.one_hot(torch.from_numpy(labels), len(data.CONFIGS)).float()
    xq, xt = XQ[train], XT[train]
    torch.manual_seed(seed)
    model = MLPRouter(XQ.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    order = torch.Generator().manual_seed(seed)
    started = time.time()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Y), generator=order)
        for i in range(0, len(Y), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.binary_cross_entropy_with_logits(model(xq[idx], xt[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_s = time.time() - started
    model.eval()
    with torch.no_grad():
        train_picks = model(xq, xt).argmax(1).numpy()
        started = time.time()
        picks = model(XQ[test], XT[test]).argmax(1).numpy()
        infer_ms = (time.time() - started) / test.sum() * 1000
    return {'alpha': alpha, 'beta': beta, 'fold': fold, 'seed': seed,
            'sample_ids': [t.ids[i] for i in np.flatnonzero(test)], 'picks': picks.tolist(),
            'train_label_accuracy': float((train_picks == labels).mean()), 'final_batch_loss': float(loss),
            'train_s': round(train_s, 1), 'infer_ms_per_query': infer_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=15)
    args = ap.parse_args()
    out = data.RUNS / 'router/mlp'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({
        'router': 'mlp', 'embedding': 'all-MiniLM-L6-v2', 'hidden': HIDDEN, 'epochs': EPOCHS, 'batch': BATCH,
        'lr': LR, 'settings': data.SETTINGS, 'folds': data.FOLDS, 'seeds': args.seeds,
        'cost_normalization': 'min-max of log cost on training responses'}, indent=2) + '\n')
    jobs = [(s, f, seed) for s in data.SETTINGS for f in range(data.FOLDS) for seed in range(args.seeds)]
    print(f"{len(jobs)} training jobs on {args.workers} workers", flush=True)
    with multiprocessing.Pool(args.workers, initializer=init) as pool, open(out / 'picks.jsonl', 'w') as f:
        for i, r in enumerate(pool.imap_unordered(train_job, jobs), 1):
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {i}/{len(jobs)} ({r['alpha']},{r['beta']}) fold {r['fold']} seed "
                  f"{r['seed']}: train label acc {r['train_label_accuracy']:.2f}, {r['train_s']} s", flush=True)


if __name__ == '__main__':
    main()
