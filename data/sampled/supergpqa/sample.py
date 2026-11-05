"""Sample six SuperGPQA fields, with 50 middle and 50 hard questions each."""
import collections
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
SEED = 224
FIELDS = ['Physics', 'Mathematics', 'Chemistry', 'Clinical Medicine', 'Biology', 'Mechanics']


def main():
    source = ROOT / 'data/raw/supergpqa/SuperGPQA-all.jsonl'
    rows = [json.loads(line) for line in source.open()]
    output = ROOT / 'data/sampled/supergpqa'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': SEED, 'source_file': str(source.relative_to(ROOT)),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'policy': 'Six fields; uniformly sample 50 middle and 50 hard per field without replacement; exact question-text deduplication within each field/difficulty pool.',
                'fields': {}}
    combined = []
    for field in FIELDS:
        selected = []
        info = {}
        for difficulty in ['middle', 'hard']:
            pool = sorted((r for r in rows if r['field'] == field and r['difficulty'] == difficulty),
                          key=lambda r: r['uuid'])
            unique = {}
            for row in pool:
                unique.setdefault(row['question'], row)
            candidates = list(unique.values())
            assert len(candidates) >= 50, (field, difficulty, len(candidates))
            chosen = random.Random(f'{SEED}/{field}/{difficulty}').sample(candidates, 50)
            selected.extend(chosen)
            info[difficulty] = {'available': len(pool), 'unique_questions': len(candidates), 'sampled': 50}
        selected.sort(key=lambda r: (r['difficulty'], r['uuid']))
        assert len(selected) == len({r['uuid'] for r in selected}) == 100
        assert len({r['question'] for r in selected}) == 100
        destination = output / (field.lower().replace(' ', '_') + '.jsonl')
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
        manifest['fields'][field] = {'strata': info, 'file': str(destination.relative_to(ROOT)),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'question_ids': [r['uuid'] for r in selected]}
        combined.extend(selected)
    assert len(combined) == len({r['uuid'] for r in combined}) == 600
    assert len({r['question'] for r in combined}) == 600
    assert collections.Counter(r['difficulty'] for r in combined) == {'middle': 300, 'hard': 300}
    (output / 'all.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in combined))
    (output / 'sampling_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Verified 600 unique questions: six fields, each with 50 middle + 50 hard.')
    for field, info in manifest['fields'].items():
        print(field, info['strata'])


if __name__ == '__main__':
    main()
