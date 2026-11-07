"""Retry a run's gateway-timeout failures (504 after ~240 s of reasoning) in background mode, one at a time.

The gateway cuts every request at ~240 s, streamed ones too (tried 11-06); a background response is submitted and
polled instead (trapi.respond(background=True)), so no single request runs that long.

Meant for a handful of calls while the run itself may still be going, so it takes no deployment lock and writes to
runs/<run>/responses.retry.jsonl and failures.retry.jsonl instead of the run's own files; append responses.retry.jsonl
to responses.jsonl once run.py has exited (before rerunning it, which would otherwise retry these calls again).
Each call is tried at most twice, and the TRAPI hold is checked before each one.

    python baseline/retry_timeouts.py --run full
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
import trapi  # noqa: E402
from run import DEFAULT_SET, MODELS  # noqa: E402


def keys(path):
    return {(r['sample_id'], r['model'], r['effort']) for r in map(json.loads, open(path))} if path.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--set', default=str(DEFAULT_SET))
    ap.add_argument('--max-output-tokens', type=int, default=64_000)
    args = ap.parse_args()
    out = trapi.RUNS / args.run
    rows = {r['sample_id']: r for r in map(json.loads, open(args.set))}
    done = keys(out / 'responses.jsonl') | keys(out / 'responses.retry.jsonl')
    todo = sorted({(f['sample_id'], f['model'], f['effort']) for path in sorted(out.glob('failures*.jsonl'))
                   if path.name != 'failures.retry.jsonl'
                   for f in map(json.loads, open(path)) if '504' in f['error']} - done,
                  key=lambda k: (list(MODELS).index(k[1]), k))
    print(f"{len(todo)} gateway-timeout failures to retry: {todo}", flush=True)
    trapi.OTHER_RETRIES = 1
    clients = {}
    for sample_id, model, effort in todo:
        trapi.check_hold()
        if model not in clients:
            clients[model] = (trapi.make_client(f'effortgraph-{args.run}-retry-{model}'), trapi.Bucket(400_000))
        client, bucket = clients[model]
        row, deployment = rows[sample_id], MODELS[model]['deployment']
        prompt = prompts.build(row)
        base = {'sample_id': sample_id, 'task_group': row['task_group'], 'dataset': row['dataset'], 'rank': row['rank'],
                'model': model, 'deployment': deployment, 'effort': effort,
                'max_output_tokens': args.max_output_tokens,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'mode': 'background'}
        try:
            result = trapi.respond(client, bucket, deployment, prompt, effort, args.max_output_tokens,
                                   tag=f"{model}/{effort}/{sample_id}", background=True)
            path, record = out / 'responses.retry.jsonl', {**base, **result, 'ts': time.time()}
            print(f"{time.strftime('%H:%M:%S')} {model}/{effort} {sample_id}: {result['status']}, "
                  f"{result['latency_s']} s, {result['usage']['output_tokens']} output tokens", flush=True)
        except (trapi.CallFailed, trapi.Stopped) as exc:
            path, record = out / 'failures.retry.jsonl', {**base, 'error': str(exc)[:500],
                                                          'stopped': isinstance(exc, trapi.Stopped), 'ts': time.time()}
            print(f"{time.strftime('%H:%M:%S')} {model}/{effort} {sample_id}: failed: {exc}", flush=True)
        with open(path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        if trapi.STOP.is_set():
            raise SystemExit(3)


if __name__ == '__main__':
    main()
