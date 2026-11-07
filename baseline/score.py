"""Score a run's responses and summarise score, tokens and cost per configuration.

Multiple-choice rows are scored with grade_mc; LiveBench rows with LiveBench's own code (grade_livebench). LiveBench
scores are cached by response id in runs/<run>/lb_scores.jsonl (coding runs the tests, AMPS_Hard may ask gpt-5-mini),
and an answer whose scoring raised is not cached, so the next scoring run retries it. AMPS_Hard's gpt-5-mini
tiebreaker goes over TRAPI; run this only when no run is using gpt-5-mini, or pass --no-llm-equiv to leave those
answers unscored for now.

    python baseline/score.py --run pilot
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_livebench  # noqa: E402
import grade_mc  # noqa: E402
import trapi  # noqa: E402
from run import DEFAULT_SET, EFFORTS, MODELS  # noqa: E402

EQUIV_MODEL = 'gpt-5-mini'


def cost(model, usage):
    p_in, p_cached, p_out = MODELS[model]['price']
    cached = (usage.get('input_tokens_details') or {}).get('cached_tokens') or 0
    return ((usage['input_tokens'] - cached) * p_in + cached * p_cached + usage['output_tokens'] * p_out) / 1e6


def score_livebench(out, rows, responses, llm_equiv):
    cache_path = out / 'lb_scores.jsonl'
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            c = json.loads(line)
            cache[c['response_id']] = c
    if llm_equiv:
        deployment = MODELS[EQUIV_MODEL]['deployment']
        trapi.check_hold()
        lock = trapi.Lock(deployment, f'score-{out.name}').__enter__()
        client, bucket = trapi.make_client(f'effortgraph-score-{out.name}'), trapi.Bucket(100_000)
        grade_livebench.use_trapi_equivalence(
            lambda p: trapi.respond(client, bucket, deployment, p, 'medium', 16_000, tag='amps-equiv'),
            out / 'amps_equiv_calls.jsonl')
    new, errors = [], []
    todo = [r for r in responses if r['dataset'] == 'livebench' and r['response_id'] not in cache]
    instruction = collections.defaultdict(list)
    for r in todo:
        record = rows[r['sample_id']]['record']
        label = f"{r['model']}-{r['effort']}"
        if record['category'] == 'instruction_following':
            instruction[(label, record['task'])].append(r)
            continue
        if record['task'] == 'AMPS_Hard' and not llm_equiv:
            score, error = amps_sympy_only(record, r['text'] or '', label)
        else:
            score, error = grade_livebench.score_one(record, r['text'] or '', label)
        (errors if error else new).append({'response_id': r['response_id'], 'score': score, 'eval_error': error})
    for (label, task), rs in instruction.items():
        scores = grade_livebench.score_instruction_following(
            [(rows[r['sample_id']]['record'], r['text'] or '') for r in rs], label, out / 'lb_work')
        for r in rs:
            new.append({'response_id': r['response_id'], 'score': scores[rows[r['sample_id']]['record']['question_id']],
                        'eval_error': None})
    with open(cache_path, 'a') as f:
        for c in new:
            f.write(json.dumps(c) + '\n')
            cache[c['response_id']] = c
    if errors:
        print(f"{len(errors)} LiveBench answers raised while scoring (left unscored, retried next time); first: "
              f"{errors[0]['eval_error']}")
    return cache


def amps_sympy_only(record, text, label):
    """AMPS_Hard with sympy only: a sympy match scores 1; anything that would need the LLM tiebreaker is an error."""
    score, error = grade_livebench.score_one(record, text, label)
    if error and 'OPENAI_API_KEY' in error:
        return 0.0, 'needs the gpt-5-mini tiebreaker (--no-llm-equiv)'
    return score, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--set', default=str(DEFAULT_SET))
    ap.add_argument('--no-llm-equiv', action='store_true', help="don't call gpt-5-mini for AMPS_Hard tiebreaks")
    args = ap.parse_args()
    out = trapi.RUNS / args.run
    rows = {}
    for line in open(args.set):
        r = json.loads(line)
        rows[r['sample_id']] = r
    responses = [json.loads(line) for line in open(out / 'responses.jsonl')]
    lb = score_livebench(out, rows, responses, not args.no_llm_equiv)
    scored = []
    for r in responses:
        row = rows[r['sample_id']]
        if row['dataset'] == 'livebench':
            score, pred = (lb[r['response_id']]['score'] if r['response_id'] in lb else None), None
        else:
            score, pred = grade_mc.grade(row, r['text'] or '')
        scored.append({k: r[k] for k in ('sample_id', 'task_group', 'dataset', 'model', 'effort', 'response_id')} | {
            'score': score, 'pred': pred, 'truncated': r['status'] != 'completed',
            'incomplete_reason': r.get('incomplete_reason'),
            'input_tokens': r['usage']['input_tokens'], 'output_tokens': r['usage']['output_tokens'],
            'reasoning_tokens': r['usage']['output_tokens_details']['reasoning_tokens'],
            'cost_usd': cost(r['model'], r['usage']), 'latency_s': r['latency_s']})
    with open(out / 'scores.jsonl', 'w') as f:
        for s in scored:
            f.write(json.dumps(s) + '\n')

    cells = collections.defaultdict(list)
    for s in scored:
        cells[(s['dataset'], s['model'], s['effort'])].append(s)
    print(f"{'dataset':10s} {'config':18s} {'n':>4s} {'score':>6s} {'unsc':>4s} {'trunc':>5s} {'out tok':>8s} "
          f"{'max out':>8s} {'$/q':>9s}")
    for dataset in ('mmlu_pro', 'supergpqa', 'livebench'):
        for model in MODELS:
            for effort in EFFORTS:
                cell = cells.get((dataset, model, effort))
                if not cell:
                    continue
                vals = [s['score'] for s in cell if s['score'] is not None]
                acc = f"{sum(vals) / len(vals):.3f}" if vals else '  n/a'
                outs = [s['output_tokens'] for s in cell]
                print(f"{dataset:10s} {model + '/' + effort:18s} {len(cell):4d} {acc:>6s} {len(cell) - len(vals):4d} "
                      f"{sum(s['truncated'] for s in cell):5d} {sum(outs) / len(outs):8.0f} {max(outs):8d} "
                      f"{sum(s['cost_usd'] for s in cell) / len(cell):9.5f}")
    print(f"total cost at list prices: ${sum(s['cost_usd'] for s in scored):.4f} over {len(scored)} calls")


if __name__ == '__main__':
    main()
