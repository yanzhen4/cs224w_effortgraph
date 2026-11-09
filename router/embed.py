"""Text embeddings of the queries and task groups (all-MiniLM-L6-v2, 384-d, unit length), shared by every router.

A query is embedded from its content, not the full prompt: the benchmark's instruction header is the same for every
question of a data set and would take ~50 of MiniLM's 256 tokens. Multiple-choice: the question and its lettered
options; LiveBench: the question's turn. Longer texts are truncated by the model. A task group is embedded from its
name, e.g. "SuperGPQA: Clinical Medicine".

    python router/embed.py      # writes data/embeddings/all-MiniLM-L6-v2.npz
"""
import hashlib
import json
import time

import numpy as np

from data import DEFAULT_SET, ROOT

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
OUT = ROOT / 'data/embeddings/all-MiniLM-L6-v2.npz'
DATASET_NAMES = {'mmlu_pro': 'MMLU-Pro', 'supergpqa': 'SuperGPQA', 'livebench': 'LiveBench'}


def query_text(row):
    rec = row['record']
    if row['dataset'] == 'livebench':
        return rec['turns'][0]
    return rec['question'] + '\n' + '\n'.join(f'{chr(65 + i)}. {o}' for i, o in enumerate(rec['options']))


def task_text(group):
    dataset, name = group.split('/', 1)
    return f"{DATASET_NAMES[dataset]}: {name.replace('_', ' ')}"


def main():
    from sentence_transformers import SentenceTransformer
    rows = [json.loads(line) for line in open(DEFAULT_SET)]
    model = SentenceTransformer(MODEL, device='cpu')
    texts = [query_text(r) for r in rows]
    started = time.time()
    query = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    per_query_ms = (time.time() - started) / len(texts) * 1000
    groups = sorted({r['task_group'] for r in rows})
    task = model.encode([task_text(g) for g in groups], normalize_embeddings=True, convert_to_numpy=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ids=np.array([r['sample_id'] for r in rows]), query=query.astype(np.float32),
             groups=np.array(groups), task=task.astype(np.float32), model=MODEL, max_seq_length=model.max_seq_length,
             per_query_ms=per_query_ms, text_sha256=hashlib.sha256('\n\0'.join(texts).encode()).hexdigest())
    print(f"{len(rows)} queries, {len(groups)} task groups -> {OUT} ({per_query_ms:.1f} ms per query on CPU)")


if __name__ == '__main__':
    main()
