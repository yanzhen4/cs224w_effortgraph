"""Merge the three selected datasets without discarding original fields."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    combined = []
    for dataset in ['livebench', 'supergpqa', 'mmlu_pro']:
        source = ROOT / 'sampled' / dataset / 'all.jsonl'
        rows = [json.loads(line) for line in source.open()]
        assert len(rows) == 600, (dataset, len(rows))
        for row in rows:
            original_id = row['uuid' if dataset == 'supergpqa' else 'question_id']
            group = row['field' if dataset == 'supergpqa' else 'category']
            combined.append({'sample_id': f'{dataset}:{original_id}',
                             'dataset': dataset, 'source_id': original_id,
                             'task_group': f'{dataset}/{group}', 'record': row})
    assert len(combined) == len({r['sample_id'] for r in combined}) == 1800
    assert len({r['task_group'] for r in combined}) == 17
    destination = ROOT / 'combined_1800.jsonl'
    with destination.open('w') as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(dict(collections.Counter(r['dataset'] for r in combined)))
    print('Wrote 1800 unique sample IDs across 17 task groups:', destination)


if __name__ == '__main__':
    main()
