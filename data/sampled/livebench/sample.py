"""Sample 100 records per category, balancing tasks and native difficulty strata."""
import collections
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
SEED = 224
TARGET = 100


def difficulty(row):
    category = row['category']
    field = {'coding': 'original_json.difficulty', 'math': 'hardness',
             'reasoning': 'level'}.get(category)
    value = ((row.get('original_json') or {}).get('difficulty') if category == 'coding'
             else row.get(field) if field else None)
    return field, 'unknown' if value is None or value == '' else str(value)


def allocate(capacities, target):
    """Equal allocation with capacity limits; deterministic remainder allocation."""
    assert sum(capacities.values()) >= target
    assert target >= len(capacities), 'Too few slots to cover every stratum'
    counts = dict.fromkeys(sorted(capacities), 0)
    while sum(counts.values()) < target:
        for key in counts:
            if counts[key] < capacities[key]:
                counts[key] += 1
                if sum(counts.values()) == target:
                    break
    return counts


def main():
    output = ROOT / 'data/sampled/livebench'
    output.mkdir(parents=True, exist_ok=True)
    combined = []
    report = {'seed': SEED, 'per_category': TARGET,
              'policy': 'Historical test pools, including removed questions; exact-prompt deduplication within category; balance task then native difficulty subject to availability. No inferred difficulty.',
              'categories': {}}
    for source in sorted((ROOT / 'data/raw/livebench').glob('*/jsonl/test*.jsonl')):
        rows = [json.loads(line) for line in source.open()]
        category = source.parent.parent.name
        unique = {}
        # When identical prompts recur, retain the newest release deterministically.
        for row in sorted(rows, key=lambda r: (str(r.get('livebench_release_date') or ''), str(r['question_id'])), reverse=True):
            prompt = json.dumps(row['turns'], ensure_ascii=False, sort_keys=True)
            unique.setdefault(prompt, row)
        tasks = collections.defaultdict(list)
        for row in unique.values():
            tasks[row['task']].append(row)
        task_quotas = allocate({t: len(v) for t, v in tasks.items()}, TARGET)
        selected = []
        task_report = {}
        for task, quota in task_quotas.items():
            strata = collections.defaultdict(list)
            for row in tasks[task]:
                strata[difficulty(row)[1]].append(row)
            quotas = allocate({d: len(v) for d, v in strata.items()}, quota)
            for label, count in quotas.items():
                candidates = sorted(strata[label], key=lambda r: str(r['question_id']))
                rng = random.Random(f'{SEED}/{category}/{task}/{label}')
                for row in rng.sample(candidates, count):
                    field, value = difficulty(row)
                    selected.append({**row, '_sampling': {'seed': SEED,
                        **({'difficulty_source': field, 'difficulty_label': value}
                           if value != 'unknown' else {}),
                        'source_file': str(source.relative_to(ROOT)),
                        'historical_pool': True}})
            task_report[task] = {'available': len(tasks[task]), 'sampled': quota,
                                 'difficulty_available': {d: len(v) for d, v in sorted(strata.items()) if d != 'unknown'},
                                 'difficulty_sampled': {d: n for d, n in quotas.items() if d != 'unknown'}}
        selected.sort(key=lambda r: (r['task'], str(r['question_id'])))
        assert len(selected) == TARGET
        assert len({r['question_id'] for r in selected}) == TARGET
        assert len({json.dumps(r['turns'], sort_keys=True) for r in selected}) == TARGET
        # Explicitly check that every original task/difficulty combination survives.
        assert {(r['task'], difficulty(r)[1]) for r in rows} <= {
            (r['task'], difficulty(r)[1]) for r in selected}
        destination = output / f'{category}.jsonl'
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
        report['categories'][category] = {'raw_records': len(rows), 'unique_prompts': len(unique),
            'sampled': len(selected), 'tasks': task_report,
            'difficulty_sampled': dict(collections.Counter(difficulty(r)[1] for r in selected if difficulty(r)[1] != 'unknown')),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sample_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
        combined.extend(selected)
    assert len(combined) == 600
    (output / 'all.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in combined))
    (output / 'sampling_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for category, info in report['categories'].items():
        print(category, 'unique:', info['unique_prompts'], 'tasks:',
              {t: v['sampled'] for t, v in info['tasks'].items()},
              'difficulty:', info['difficulty_sampled'])


if __name__ == '__main__':
    main()
