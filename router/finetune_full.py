"""CPU MiniLM full fine-tuning, starting from the matching balance MLP checkpoint."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / '.cache'
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))
os.environ.setdefault('TORCH_HOME', str(CACHE / 'torch'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import transformers

import data
import embed
import mlp


class FullRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(embed.MODEL, local_files_only=True,
                                                add_pooling_layer=False, attn_implementation='sdpa')
        self.router = mlp.MLPRouter(384)

    def encode(self, batch):
        states = self.encoder(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).to(states.dtype)
        return F.normalize((states * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1)

    def forward(self, queries, tasks, task_index):
        return self.router(self.encode(queries), self.encode(tasks)[task_index])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accumulation', type=int, default=4)
    ap.add_argument('--encoder-lr', type=float, default=2e-5)
    ap.add_argument('--head-lr', type=float, default=1e-4)
    ap.add_argument('--setting', choices=('quality', 'balanced', 'cost'), default='quality')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--init', choices=('warm', 'scratch'), default='warm')
    ap.add_argument('--save-every', type=int, default=0, help='save weights every N epochs, plus final epoch')
    args = ap.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t = data.load()
    alpha, beta = next(pair for pair, name in mlp.SETTING_NAMES.items() if name == args.setting)
    train, test, labels = mlp.training_split(t, alpha, beta, args.fold, ROOT / 'data/sampled_v2_balance')
    train_idx, test_idx = np.flatnonzero(train), np.flatnonzero(test)
    train_label = dict(zip(train_idx.tolist(), labels.tolist()))
    out = args.output or data.RUNS / 'router' / f'full_balance_{args.setting}_fold{args.fold}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in data.DEFAULT_SET.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(embed.MODEL, local_files_only=True)
    groups = sorted(set(t.group))
    group_ids = [groups.index(g) for g in t.group]
    # Tokenization is cached; encoder activations are recomputed during training.
    token_path = CACHE / 'effortgraph/minilm_tokens_800.json'
    if token_path.exists():
        tokens = json.loads(token_path.read_text())
        assert tokens['ids'] == t.ids and tokens['groups'] == groups
        query_tokens, task_tokens = tokens['query'], tokens['task']
    else:
        query_tokens = tokenizer([embed.query_text(q) for q in rows], truncation=True, max_length=256)
        task_tokens = tokenizer([embed.task_text(g) for g in groups], truncation=True, max_length=256)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = token_path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps({'ids': t.ids, 'groups': groups, 'query': dict(query_tokens),
                                         'task': dict(task_tokens)}) + '\n')
        temporary.replace(token_path)

    def batch(indices):
        tasks = sorted({group_ids[i] for i in indices})
        q = tokenizer.pad([{k: v[i] for k, v in query_tokens.items()} for i in indices], return_tensors='pt')
        tg = tokenizer.pad([{k: v[i] for k, v in task_tokens.items()} for i in tasks], return_tensors='pt')
        return q, tg, torch.tensor([tasks.index(group_ids[i]) for i in indices])

    model = FullRouter()
    warm_path = data.RUNS / 'router/mlp_balance/checkpoints' / f'{args.setting}_fold{args.fold}_seed{args.seed}.pt'
    warm = torch.load(warm_path, weights_only=True)
    assert warm['train_ids'] == [t.ids[i] for i in train_idx]
    if args.init == 'warm':
        model.router.load_state_dict(warm['state_dict'])
    assert all(p.requires_grad for p in model.parameters())
    initial_weight = model.encoder.encoder.layer[0].attention.self.query.weight.detach().clone()
    model.eval()
    cached = np.load(ROOT / 'data/embeddings/all-MiniLM-L6-v2.npz')
    with torch.no_grad():
        q, tg, ti = batch(train_idx[:8].tolist())
        live = model.encode(q)
        embedding_error = float((live - torch.from_numpy(cached['query'][train_idx[:8]])).abs().max())
    assert embedding_error < 1e-4, f'Embedding mismatch: {embedding_error}'
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': args.encoder_lr},
                                   {'params': model.router.parameters(), 'lr': args.head_lr}], weight_decay=0.01)
    config = {**vars(args), 'output': str(out), 'alpha': alpha, 'beta': beta, 'device': 'cpu',
              'warm_start': str(warm_path.relative_to(ROOT)) if args.init == 'warm' else None, 'encoder': embed.MODEL,
              'encoder_revision': Path(hf_hub_download(embed.MODEL, 'config.json', local_files_only=True)).parent.name,
              'parameters': sum(p.numel() for p in model.parameters()),
              'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
              'train_questions': len(train_idx), 'test_questions': len(test_idx),
              'effective_batch': args.batch * args.accumulation, 'optimizer': 'AdamW',
              'weight_decay': 0.01, 'gradient_clip': 1.0, 'loss': 'BCEWithLogits',
              'selection': 'fixed epochs, final checkpoint; no test-based selection',
              'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'transformers': transformers.__version__},
              'initial_embedding_max_error': embedding_error}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f"Full fine-tuning: {config['parameters']:,} trainable parameters; {len(train_idx)} train / {len(test_idx)} test", flush=True)
    setup_s = time.perf_counter() - started
    history = []
    def save_checkpoint(epoch):
        destination = out / (f'checkpoint_epoch_{epoch:03d}.pt' if args.save_every else 'checkpoint.pt')
        temporary = destination.with_suffix('.tmp')
        torch.save({'encoder_state_dict': model.encoder.state_dict(), 'router_state_dict': model.router.state_dict(),
                    'config': config, 'epoch': epoch, 'train_ids': warm['train_ids']}, temporary)
        temporary.replace(destination)

    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        perm = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        losses, correct = [], 0
        optimizer.zero_grad(set_to_none=True)
        for step, offset in enumerate(range(0, len(perm), args.batch)):
            idx = perm[offset:offset + args.batch].tolist()
            target = torch.tensor([train_label[i] for i in idx])
            logits = model(*batch(idx))
            loss = F.binary_cross_entropy_with_logits(logits, F.one_hot(target, 9).float())
            (loss / args.accumulation).backward()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(1) == target).sum())
            if (step + 1) % args.accumulation == 0 or offset + args.batch >= len(perm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        record = {'epoch': epoch + 1, 'seconds': time.perf_counter() - epoch_start,
                  'train_loss': float(np.mean(losses)), 'train_label_accuracy_online': correct / len(train_idx)}
        history.append(record)
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f"Epoch {epoch+1}/{args.epochs}: {record['seconds']:.1f}s, loss={record['train_loss']:.4f}, train label acc={correct/len(train_idx):.3f}", flush=True)
        if args.save_every and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            save_checkpoint(epoch + 1)
    model.eval()
    eval_start = time.perf_counter()
    picks = []
    with torch.no_grad():
        for offset in range(0, len(test_idx), args.batch):
            picks.extend(model(*batch(test_idx[offset:offset+args.batch].tolist())).argmax(1).tolist())
    eval_s = time.perf_counter() - eval_start
    weight_delta = float((model.encoder.encoder.layer[0].attention.self.query.weight.detach() - initial_weight).abs().max())
    assert weight_delta > 0
    p, c = t.P[test_idx, picks], t.C[test_idx, picks]
    norm = data.Normalizer(t.P[t.fold != args.fold], t.C[t.fold != args.fold])
    baseline_records = [json.loads(l) for l in (data.RUNS / 'router/mlp_balance/picks.jsonl').read_text().splitlines()]
    baseline = next(r for r in baseline_records if (r['alpha'], r['beta'], r['fold'], r['seed']) == (alpha, beta, args.fold, args.seed))
    assert baseline['sample_ids'] == [t.ids[i] for i in test_idx]
    bp, bc = t.P[test_idx, baseline['picks']], t.C[test_idx, baseline['picks']]
    metrics = lambda p, c: {'score': float(p.mean()), 'cost': float(c.mean()),
                            'reward': float(norm.reward(p, c, alpha, beta).mean())}
    if not args.save_every:
        save_checkpoint(args.epochs)
    result = {'alpha': alpha, 'beta': beta, 'fold': args.fold, 'seed': args.seed,
              'sample_ids': [t.ids[i] for i in test_idx], 'picks': picks,
              'baseline': metrics(bp, bc), 'full_finetune': metrics(p, c),
              'setup_s': setup_s, 'training_s': sum(r['seconds'] for r in history),
              'test_inference_s': eval_s, 'total_s': time.perf_counter() - started,
              'encoder_query_weight_max_change': weight_delta}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('picks', 'sample_ids')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
