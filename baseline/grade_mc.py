"""Answer extraction for the two multiple-choice benchmarks, copied from their evaluation code.

MMLU-Pro: extract_answer / extract_again / extract_final from evaluate_from_api.py (f418b11), after the same
response.replace('**', ''). The official script scores an unparsed answer by a seeded random guess; here it scores 0
and is flagged, so a label never depends on a guess.
SuperGPQA: extract_option_labels, then extract_option_content, from eval/eval.py (e31e8bf), zero-shot mode.
The patterns run under a 5 s alarm, as eval.py's timeout_decorator does (main thread only).
"""
import re
import signal


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def safe_regex_search(pattern, text, flags=0):
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(5)
    try:
        return re.search(pattern, text, flags)
    except _Timeout:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------- MMLU-Pro
def mmlu_extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return mmlu_extract_again(text)


def mmlu_extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    return mmlu_extract_final(text)


def mmlu_extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return None


# ---------------------------------------------------------------- SuperGPQA
# Verbatim from SuperGPQA eval/eval.py (e31e8bf).
def extract_option_labels(text, options='ABCDEFGHIJ'):
    if not isinstance(text, str) or not isinstance(options, str):
        return 'error'
    
    text = text.rstrip()
    last_line = text.split('\n')[-1]
    
    option_str = ''.join([chr(65 + i) for i in range(len(options))]) if options else 'ABCDEFGHIJ'
    
    patterns = [
        # e.g. "The final answer to this question is: A."
        #      "The best option is $\boxed{B}:"
        #      "The correct answer is (C)."
        f'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',

        # e.g. "ANSWER: A"
        #      "Answer: $\boxed{B}."
        #      "ANSWER: (C):"
        f'(?i:Answer)[\*\s]*:\s*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        
        # e.g. "A"
        #      "$\boxed{B}$"
        #      "(C)."
        #      "[D]:"
        f'^[^\w\r\n]*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
    ]

    for pattern in patterns:
        match = safe_regex_search(pattern, last_line, re.IGNORECASE)
        if match:
            return match.group(1)
    
    for pattern in patterns:
        match = safe_regex_search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def extract_option_content(text, options_content=None):
    if not isinstance(text, str) or not isinstance(options_content, list):
        return 'error'
    
    escaped_options_content = [re.escape(option_content) for option_content in options_content]
    escaped_options_content_str = '|'.join(escaped_options_content)
    
    text = text.rstrip()
    last_line = text.split('\n')[-1]
        
    patterns = [
        f'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\\{{\(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*({escaped_options_content_str})(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        
        f'(?i:Answer)\s*(?:[\*\$\\{{\(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*({escaped_options_content_str})(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        
        f'^[^\w\r\n]*(?:[\*\$\\{{\(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*({escaped_options_content_str})(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
    ]
    
    for pattern in patterns:
        match = safe_regex_search(pattern, last_line)
        if match:
            if match.group(1) in escaped_options_content:
                return options_content[escaped_options_content.index(match.group(1))]
            else:
                return match.group(1)
    
    for pattern in patterns:
        match = safe_regex_search(pattern, text)
        if match:
            if match.group(1) in escaped_options_content:
                return options_content[escaped_options_content.index(match.group(1))]
            else:
                return match.group(1)
    
    return None


def supergpqa_predict(response, options):
    predict = extract_option_labels(response, 'ABCDEFGHIJ')
    if predict is None:
        predict = extract_option_content(response, options)
        predict = chr(options.index(predict) + 65) if predict else None
    return predict


def grade(row, text):
    """(score, extracted answer) for a multiple-choice row; score is 1.0 or 0.0."""
    record = row['record']
    if row['dataset'] == 'mmlu_pro':
        pred = mmlu_extract_answer(text.replace('**', ''))
        return float(pred == record['answer']), pred
    if row['dataset'] == 'supergpqa':
        pred = supergpqa_predict(text, record['options'])
        return float(pred == record['answer_letter']), pred
    raise ValueError(row['dataset'])
