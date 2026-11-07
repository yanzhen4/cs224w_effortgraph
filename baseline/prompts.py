"""Zero-shot prompts, each as the benchmark's own evaluation code builds it.

MMLU-Pro: evaluate_from_api.py (TIGER-AI-Lab/MMLU-Pro f418b11) without the five worked examples.
SuperGPQA: config/prompt/zero-shot.yaml and infer/data_loader.py (SuperGPQA/SuperGPQA e31e8bf).
LiveBench: the question's single turn, as gen_api_answer.py sends it (GPT-5 configs have no prompt prefix).
"""

MMLU_PRO_HEADER = ('The following are multiple choice questions (with answers) about {}. Think step by'
                   ' step and then output the answer in the format of "The answer is (X)" at the end.\n\n')
SUPERGPQA_TEMPLATE = ("Answer the following multiple choice question. There is only one correct answer. The last line of "
                      "your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of "
                      "A, B, C, D, E, F, G, H, I, or J.\n\n{}\n")


def mmlu_pro_example(question, options, cot_content=""):
    # format_example() from MMLU-Pro's evaluate_from_api.py, unchanged.
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def build(row):
    record = row['record']
    if row['dataset'] == 'mmlu_pro':
        return MMLU_PRO_HEADER.format(record['category']) + mmlu_pro_example(record['question'], record['options'])
    if row['dataset'] == 'supergpqa':
        body = record['question'] + '\n' + '\n'.join(f'{chr(65 + i)}) {option}' for i, option in enumerate(record['options']))
        return SUPERGPQA_TEMPLATE.format(body)
    if row['dataset'] == 'livebench':
        assert len(record['turns']) == 1 and 'system_prompt' not in record
        return record['turns'][0]
    raise ValueError(row['dataset'])
