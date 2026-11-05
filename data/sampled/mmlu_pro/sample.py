"""Sample 600 MMLU-Pro test questions, balancing sources within each category."""
import collections
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
SEED = 224
TARGETS = {'physics': 200, 'math': 100, 'chemistry': 100, 'health': 100, 'biology': 100}


def allocate(capacities, target):
    assert sum(capacities.values()) >= target >= len(capacities)
    quotas = dict.fromkeys(sorted(capacities), 0)
    while sum(quotas.values()) < target:
        for key in quotas:
            if quotas[key] < capacities[key]:
                quotas[key] += 1
                if sum(quotas.values()) == target:
                    break
    return quotas


def main():
    source = ROOT / 'data/raw/mmlu_pro/jsonl/test-00000-of-00001.jsonl'
    rows = [json.loads(line) for line in source.open()]
    output = ROOT / 'data/sampled/mmlu_pro'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': SEED, 'source_file': str(source.relative_to(ROOT)),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'policy': 'Test split only; physics 200, math/chemistry/health/biology 100 each. Balance src within category, with deterministic remainder allocation. Exact question-text deduplication before sampling. No difficulty labels inferred.',
                'categories': {}}
    combined = []
    for category, target in TARGETS.items():
        pool = sorted((r for r in rows if r['category'] == category), key=lambda r: r['question_id'])
        unique = {}
        for row in pool:
            unique.setdefault(row['question'], row)
        strata = collections.defaultdict(list)
        for row in unique.values():
            strata[row['src']].append(row)
        quotas = allocate({s: len(v) for s, v in strata.items()}, target)
        selected = []
        for src, count in quotas.items():
            selected.extend(random.Random(f'{SEED}/{category}/{src}').sample(strata[src], count))
        selected.sort(key=lambda r: r['question_id'])
        assert len(selected) == len({r['question_id'] for r in selected}) == target
        assert {r['src'] for r in selected} == {r['src'] for r in pool}
        destination = output / f'{category}.jsonl'
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
        manifest['categories'][category] = {'available': len(pool), 'unique_questions': len(unique),
            'sampled': len(selected), 'sources': {s: {'available': len(strata[s]), 'sampled': q} for s, q in quotas.items()},
            'file': str(destination.relative_to(ROOT)),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'question_ids': [r['question_id'] for r in selected]}
        combined.extend(selected)
    assert len(combined) == len({r['question_id'] for r in combined}) == 600
    assert len({r['question'] for r in combined}) == 600
    assert collections.Counter(r['category'] for r in combined) == TARGETS
    (output / 'all.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in combined))
    (output / 'sampling_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for category, info in manifest['categories'].items():
        print(category, info['sampled'], 'questions;', len(info['sources']), 'sources covered')
    print('Verified 600 unique IDs and question texts.')


if __name__ == '__main__':
    main()
