"""LiveBench scoring with LiveBench's own code (third_party/LiveBench 1d1006a, pip-installed in env effortgraph).

Every task except instruction following goes through gen_ground_truth_judgment.play_a_match_gt, one answer at a time,
as gen_judgments does; a scoring-time exception scores 0 there and is kept here as eval_error.
Instruction following (2024 format) goes through instruction_following_process_results, one batch per configuration,
as gen_judgments' old-format path does; its evaluator writes logs under the working directory, so it runs in workdir.
"""
import contextlib
import io
import json
import os
import re
import time

_nltk_ready = False
EQUIV_PROMPT = 'Are these latex expressions equivalent or negatives of each other. Reply yes or no: \n "{}" and "{}"'


def use_trapi_equivalence(respond, log_path):
    """Route AMPS_Hard's LLM tiebreaker to TRAPI gpt-5-mini.

    amps_hard_process_results asks gpt-5-mini (OpenAI client, default effort) whether two expressions are equivalent
    whenever sympy does not confirm it, and refuses to run without OPENAI_API_KEY. This swaps in the same prompt and
    the same 'starts with yes' rule over TRAPI; the placeholder key only satisfies that guard and is never sent.
    respond(prompt) -> response dict from trapi.respond.
    """
    from livebench.process_results.math.AMPS_Hard import utils as amps

    def is_equiv_llm(x1, x2):
        prompt = EQUIV_PROMPT.format(x1, x2)
        result = respond(prompt)
        with open(log_path, 'a') as f:
            f.write(json.dumps({'ts': time.time(), 'x1': x1, 'x2': x2, 'text': result['text'],
                                'usage': result['usage']}) + '\n')
        return (result['text'] or '').strip().lower().startswith('yes')

    amps.is_equiv_llm = is_equiv_llm
    os.environ.setdefault('OPENAI_API_KEY', 'unused-routed-to-trapi')


def score_one(record, text, label):
    """(score, eval_error) for one non-instruction-following answer."""
    from livebench.common import MatchSingle
    from livebench.gen_ground_truth_judgment import play_a_match_gt
    assert record['category'] != 'instruction_following'
    match = MatchSingle(question=record, model=label, answer={'choices': [{'turns': [text]}]})
    with contextlib.redirect_stdout(io.StringIO()):
        result = play_a_match_gt(match, output_file=None)
    return float(result['score']), result.get('error_msg')


def score_instruction_following(items, label, workdir):
    """{question_id: score} for a batch [(record, text)] answered by one configuration."""
    global _nltk_ready
    import nltk
    from livebench.process_results.instruction_following.utils import instruction_following_process_results
    if not _nltk_ready:
        for package in ('punkt', 'punkt_tab', 'averaged_perceptron_tagger'):
            nltk.download(package, quiet=True)
        _nltk_ready = True
    questions = [record for record, _ in items]
    assert all(q['category'] == 'instruction_following' and q['livebench_release_date'] < '2025-11-25' for q in questions)
    answers = {label: {record['question_id']: {
        'question_id': record['question_id'],
        'choices': [{'turns': [re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()]}]}
        for record, text in items}}
    # instruction_following_process_results logs to data/live_bench/instruction_following/<task>/model_judgment.
    os.makedirs(os.path.join(workdir, f"data/live_bench/instruction_following/{questions[0]['task']}/model_judgment"),
                exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scores = instruction_following_process_results(questions, answers, questions[0]['task'], label)
    finally:
        os.chdir(cwd)
    return {s['question_id']: float(s['score']) for s in scores}
