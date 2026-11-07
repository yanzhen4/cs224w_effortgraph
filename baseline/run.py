"""Collect baseline responses: each selected question under every model x reasoning-effort configuration, once.

One process, one thread pool per model deployment (each has its own quota, lock and token bucket). Responses are
appended to runs/<run>/responses.jsonl as they finish; a rerun skips configurations already answered, so failed calls
are simply retried by running the same command again. Exit status 3 means a TRAPI block or the 429 breaker stopped it.

    python baseline/run.py --run pilot --pilot            # first 10 questions of each group
    python baseline/run.py --run smoke --per-group 1 --models gpt-5-nano --efforts low
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
import trapi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / 'data/combined_v2_800.jsonl'
# Prices per 1M tokens as LiveBench's model configs record them (input, cached input, output incl. reasoning).
MODELS = {
    'gpt-5-nano': {'deployment': 'gpt-5-nano_2025-08-07', 'price': (0.05, 0.005, 0.40)},
    'gpt-5-mini': {'deployment': 'gpt-5-mini_2025-08-07', 'price': (0.25, 0.025, 2.00)},
    'gpt-5': {'deployment': 'gpt-5_2025-08-07', 'price': (1.25, 0.125, 10.00)},
}
EFFORTS = ['low', 'medium', 'high']


def load_set(path, pilot, per_group, groups):
    rows = [json.loads(line) for line in open(path)]
    if groups:
        rows = [r for r in rows if r['task_group'] in groups]
    if pilot:
        rows = [r for r in rows if r['pilot']]
    if per_group is not None:
        rows = [r for r in rows if r['rank'] < per_group]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--set', default=str(DEFAULT_SET))
    ap.add_argument('--pilot', action='store_true', help='only questions with rank < 10')
    ap.add_argument('--per-group', type=int, help='only the first n questions of each group')
    ap.add_argument('--groups', nargs='*')
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    ap.add_argument('--efforts', nargs='*', default=EFFORTS)
    ap.add_argument('--workers', type=int, default=6, help='concurrent calls per model')
    ap.add_argument('--tpm', type=int, default=400_000, help='tokens per minute reserved per model')
    ap.add_argument('--max-output-tokens', type=int, default=64_000)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rows = load_set(args.set, args.pilot, args.per_group, args.groups)
    out = trapi.RUNS / args.run
    done = set()
    if (out / 'responses.jsonl').exists():
        for line in open(out / 'responses.jsonl'):
            r = json.loads(line)
            done.add((r['sample_id'], r['model'], r['effort']))
    jobs = {m: [(row, e) for row in sorted(rows, key=lambda r: (r['rank'], r['task_group']))
                for e in args.efforts if (row['sample_id'], m, e) not in done] for m in args.models}
    print(f"{len(rows)} questions; {sum(map(len, jobs.values()))} calls to make "
          f"({len(done)} already in {out / 'responses.jsonl'})", flush=True)
    for m, js in jobs.items():
        print(f"  {m}: {len(js)} calls", flush=True)
    if args.dry_run:
        lengths = sorted(len(prompts.build(r)) for r in rows)
        print(f"prompt chars: median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
        return

    trapi.check_hold()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'run.json'
    runs = json.loads(manifest.read_text()) if manifest.exists() else []
    runs.append({'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'argv': sys.argv,
                 'set_sha256': hashlib.sha256(Path(args.set).read_bytes()).hexdigest(),
                 'questions': len(rows), 'models': {m: MODELS[m] for m in args.models},
                 'efforts': args.efforts, 'max_output_tokens': args.max_output_tokens,
                 'workers': args.workers, 'tpm': args.tpm})
    manifest.write_text(json.dumps(runs, indent=2) + '\n')

    write_lock = threading.Lock()
    counts = {m: {'ok': 0, 'failed': 0} for m in args.models}

    def work(model, client, bucket, row, effort):
        if trapi.STOP.is_set():
            return
        prompt = prompts.build(row)
        base = {'sample_id': row['sample_id'], 'task_group': row['task_group'], 'dataset': row['dataset'],
                'rank': row['rank'], 'model': model, 'deployment': MODELS[model]['deployment'], 'effort': effort,
                'max_output_tokens': args.max_output_tokens,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            result = trapi.respond(client, bucket, MODELS[model]['deployment'], prompt, effort,
                                   args.max_output_tokens, tag=f"{model}/{effort}/{row['sample_id']}")
            path, key = out / 'responses.jsonl', 'ok'
            record = {**base, **result, 'ts': time.time()}
        except (trapi.CallFailed, trapi.Stopped) as exc:
            path, key = out / 'failures.jsonl', 'failed'
            record = {**base, 'error': str(exc)[:500], 'stopped': isinstance(exc, trapi.Stopped), 'ts': time.time()}
        with write_lock:
            with open(path, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts[model][key] += 1
            n = counts[model]['ok'] + counts[model]['failed']
            if n % 25 == 0 or n == len(jobs[model]):
                print(f"{time.strftime('%H:%M:%S')} {model}: {counts[model]} of {len(jobs[model])}", flush=True)

    def run_model(model):
        client = trapi.make_client(f'effortgraph-{args.run}-{model}')
        bucket = trapi.Bucket(args.tpm)
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            for future in [pool.submit(work, model, client, bucket, row, e) for row, e in jobs[model]]:
                future.result()

    with contextlib.ExitStack() as locks:
        for m in args.models:
            locks.enter_context(trapi.Lock(MODELS[m]['deployment'], args.run))
        threads = [threading.Thread(target=run_model, args=(m,), name=m) for m in args.models if jobs[m]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(json.dumps(counts), flush=True)
    if trapi.STOP.is_set():
        print(f'stopped: {trapi.HOLD.read_text().strip() if trapi.HOLD.exists() else "block"}', flush=True)
        raise SystemExit(3)


if __name__ == '__main__':
    main()
