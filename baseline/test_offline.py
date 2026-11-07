"""Offline checks (no API calls): prompts and answer extraction against the benchmarks' own code.

    python baseline/test_offline.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_mc  # noqa: E402
import prompts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT.parent / 'third_party'


def official(path, names, extra=None):
    """Exec the named top-level functions from an official source file (without its heavy imports)."""
    text = (THIRD / path).read_text()
    namespace = {'re': re, **(extra or {})}
    for name in names:
        m = re.search(rf'^def {name}\(.*?(?=^\S)', text, re.S | re.M)
        exec(m.group(0), namespace)
    return namespace


def main():
    rows = [json.loads(line) for line in open(ROOT / 'data/combined_v2_800.jsonl')]
    by = {d: [r for r in rows if r['dataset'] == d] for d in ('mmlu_pro', 'supergpqa', 'livebench')}

    # Prompts, character for character.
    mp = official('MMLU-Pro/evaluate_from_api.py', ['format_example'])
    for r in by['mmlu_pro']:
        rec = r['record']
        header = ('The following are multiple choice questions (with answers) about {}. Think step by'
                  ' step and then output the answer in the format of "The answer is (X)" at the end.\n\n'
                  .format(rec['category']))
        assert prompts.build(r) == header + mp['format_example'](rec['question'], rec['options']), r['sample_id']
    template = yaml.safe_load((THIRD / 'SuperGPQA/config/prompt/zero-shot.yaml').read_text())['prompt_format'][0]
    for r in by['supergpqa']:
        item = r['record']
        fmt = [item['question'] + '\n' + '\n'.join([f'{chr(65 + i)}) {option}' for i, option in enumerate(item['options'])])]
        assert prompts.build(r) == template.format(*fmt), r['sample_id']
    for r in by['livebench']:
        assert prompts.build(r) == r['record']['turns'][0]
    print('prompts match the official construction:', {d: len(v) for d, v in by.items()})

    # Extraction against the official functions.
    mo = official('MMLU-Pro/evaluate_from_api.py', ['extract_answer', 'extract_again', 'extract_final'],
                  {'print': lambda *a, **k: None})
    so = official('SuperGPQA/eval/eval.py', ['extract_option_labels', 'extract_option_content'],
                  {'safe_regex_search': lambda p, t, f=0: re.search(p, t, f)})
    samples = ['The answer is (C).', 'so the answer is D', '**Answer:** B', 'Answer: J', 'I think E is right.',
               'Final: \\boxed{G}', 'The correct option is (A)', 'no letter here', 'ANSWER: (H):', 'A',
               'Thus...\nAnswer: $\\boxed{F}$', 'the best option is $\\boxed{B}:', 'x\nAnswer: 42']
    for s in samples:
        assert grade_mc.mmlu_extract_answer(s.replace('**', '')) == mo['extract_answer'](s.replace('**', '')), s
        assert grade_mc.extract_option_labels(s) == so['extract_option_labels'](s), s
    opts = by['supergpqa'][0]['record']['options']
    for s in [f'The answer is {opts[2]}', f'Answer: {opts[5]}', 'nothing']:
        assert grade_mc.extract_option_content(s, opts) == so['extract_option_content'](s, opts), s
    print('extraction matches the official functions on', len(samples), 'samples')

    # Gold answers score 1, a wrong letter scores 0.
    for r in by['mmlu_pro']:
        gold = r['record']['answer']
        wrong = 'A' if gold != 'A' else 'B'
        assert grade_mc.grade(r, f"Let's see.\nThe answer is ({gold})") == (1.0, gold)
        assert grade_mc.grade(r, f"The answer is ({wrong})")[0] == 0.0
    for r in by['supergpqa']:
        gold = r['record']['answer_letter']
        wrong = 'A' if gold != 'A' else 'B'
        assert grade_mc.grade(r, f"Reasoning...\nAnswer: {gold}") == (1.0, gold)
        assert grade_mc.grade(r, f"Answer: {wrong}")[0] == 0.0
        assert r['record']['options'][ord(gold) - 65] == r['record']['answer']
    print('gold letters score 1 and wrong letters 0 on all', len(by['mmlu_pro']) + len(by['supergpqa']), 'MC questions')


if __name__ == '__main__':
    main()
