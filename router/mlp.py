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
import hashlib
import json
import multiprocessing
import platform
from pathlib import Path
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
BALANCE = OUTPUT = None
TRAIN_EPOCHS = EPOCHS
SETTING_NAMES = {(1.0, 0.0): 'quality', (0.5, 0.5): 'balanced', (0.2, 0.8): 'cost'}


def init(balance=None, output=None, epochs=EPOCHS):
    global TABLE, XQ, XT, BALANCE, OUTPUT, TRAIN_EPOCHS
    torch.set_num_threads(1)
    TABLE = data.load()
    XQ, XT = features(TABLE)
    BALANCE, OUTPUT, TRAIN_EPOCHS = balance, output, epochs


def training_split(t, alpha, beta, fold, balance=None):
    train, test = t.fold != fold, t.fold == fold
    if balance is None:
        R = data.Normalizer(t.P[train], t.C[train]).reward(t.P[train], t.C[train], alpha, beta)
        return train, test, data.best(R, t.C[train])
    folder = Path(balance) / SETTING_NAMES[(alpha, beta)] / f'fold_{fold}'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert (manifest['alpha'], manifest['beta'], manifest['fold']) == (alpha, beta, fold)
    ids = [json.loads(line)['sample_id'] for line in (folder / 'train_balance.jsonl').read_text().splitlines()]
    test_ids = [json.loads(line)['sample_id'] for line in (folder / 'test.jsonl').read_text().splitlines()]
    assert test_ids == [q for q, use in zip(t.ids, test) if use]
    assert len(ids) == len(set(ids)) == 320 and set(ids) <= set(t.ids)
    assert not set(ids).intersection(test_ids)
    train = np.isin(t.ids, ids)
    gold = manifest['train_gold_labels']
    assert set(gold) == set(ids)
    labels = np.array([gold[q] for q, use in zip(t.ids, train) if use], dtype=np.int64)
    assert np.bincount(labels, minlength=9).tolist() == manifest['label_counts_after']
    return train, test, labels


def train_job(job):
    (alpha, beta), fold, seed = job
    t = TABLE
    train, test, labels = training_split(t, alpha, beta, fold, BALANCE)
    Y = F.one_hot(torch.from_numpy(labels), len(data.CONFIGS)).float()
    xq, xt = XQ[train], XT[train]
    torch.manual_seed(seed)
    model = MLPRouter(XQ.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    order = torch.Generator().manual_seed(seed)
    started = time.time()
    for _ in range(TRAIN_EPOCHS):
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
    if OUTPUT:
        checkpoint = Path(OUTPUT) / 'checkpoints' / f'{SETTING_NAMES[(alpha, beta)]}_fold{fold}_seed{seed}.pt'
        torch.save({'state_dict': model.state_dict(), 'd_text': XQ.shape[1], 'hidden': HIDDEN,
                    'configs': data.CONFIGS, 'alpha': alpha, 'beta': beta, 'fold': fold, 'seed': seed,
                    'train_ids': [q for q, use in zip(t.ids, train) if use]}, checkpoint)
    return {'alpha': alpha, 'beta': beta, 'fold': fold, 'seed': seed,
            'train_questions': int(train.sum()), 'test_questions': int(test.sum()),
            'sample_ids': [t.ids[i] for i in np.flatnonzero(test)], 'picks': picks.tolist(),
            'train_label_accuracy': float((train_picks == labels).mean()), 'final_batch_loss': float(loss.detach()),
            'train_s': round(train_s, 1), 'infer_ms_per_query': infer_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=15)
    ap.add_argument('--balance', action='store_true', help='use sampled_v2_balance fixed splits and gold labels')
    ap.add_argument('--output', help='output name under runs/router')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--fold', type=int, choices=range(data.FOLDS))
    ap.add_argument('--setting', choices=SETTING_NAMES.values())
    args = ap.parse_args()
    out = data.RUNS / 'router' / (args.output or ('mlp_balance' if args.balance else 'mlp'))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    balance = str(data.ROOT / 'data/sampled_v2_balance') if args.balance else None
    (out / 'config.json').write_text(json.dumps({
        'router': 'mlp', 'embedding': 'all-MiniLM-L6-v2', 'hidden': HIDDEN, 'epochs': args.epochs, 'batch': BATCH,
        'training_data': 'sampled_v2_balance' if args.balance else 'combined_v2_800',
        'normalization_source': 'original 640-question training fold; balance labels frozen before sampling',
        'lr': LR, 'settings': data.SETTINGS, 'folds': data.FOLDS, 'seeds': args.seeds,
        'workers': args.workers, 'device': 'cpu', 'threads_per_worker': 1,
        'versions': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
        'input_sha256': {str(p.relative_to(data.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [data.ROOT / 'data/combined_v2_800.jsonl', data.RUNS / 'full/scores.jsonl',
                          data.ROOT / 'data/embeddings/all-MiniLM-L6-v2.npz']
                         + ([Path(balance) / 'sampling_manifest.json'] if balance else [])},
        'cost_normalization': 'min-max of log cost on training responses'}, indent=2) + '\n')
    jobs = [(s, f, seed) for s in data.SETTINGS for f in range(data.FOLDS) for seed in range(args.seeds)]
    jobs = [job for job in jobs if (args.fold is None or job[1] == args.fold)
            and (args.setting is None or SETTING_NAMES[job[0]] == args.setting)]
    print(f"{len(jobs)} training jobs on {args.workers} workers", flush=True)
    with multiprocessing.Pool(args.workers, initializer=init,
                              initargs=(balance, str(out), args.epochs)) as pool, open(out / 'picks.jsonl', 'w') as f:
        for i, r in enumerate(pool.imap_unordered(train_job, jobs), 1):
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {i}/{len(jobs)} ({r['alpha']},{r['beta']}) fold {r['fold']} seed "
                  f"{r['seed']}: train label acc {r['train_label_accuracy']:.2f}, {r['train_s']} s", flush=True)


if __name__ == '__main__':
    main()
