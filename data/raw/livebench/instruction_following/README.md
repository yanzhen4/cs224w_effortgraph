---
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: task
    dtype: string
  - name: turns
    sequence: string
  - name: category
    dtype: string
  - name: instruction_id_list
    sequence: string
  - name: kwargs
    list:
    - name: num_bullets
      dtype: int64
    - name: num_paragraphs
      dtype: int64
    - name: num_words
      dtype: int64
    - name: relation
      dtype: string
    - name: forbidden_words
      sequence: string
    - name: section_spliter
      dtype: string
    - name: num_sections
      dtype: int64
    - name: keywords
      sequence: string
    - name: end_phrase
      dtype: string
    - name: postscript_marker
      dtype: string
    - name: num_sentences
      dtype: int64
    - name: nth_paragraph
      dtype: int64
    - name: first_word
      dtype: string
    - name: prompt_to_repeat
      dtype: string
  - name: task_prompt
    dtype: string
  - name: livebench_release_date
    dtype: timestamp[s]
  - name: livebench_removal_date
    dtype: string
  - name: release_date
    dtype: string
  - name: citation
    dtype: string
  splits:
  - name: test
    num_bytes: 1036075
    num_examples: 400
  download_size: 537024
  dataset_size: 1036075
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
arxiv: 2406.19314
---

# Dataset Card for "livebench/instruction_following"

LiveBench is a benchmark for LLMs designed with test set contamination and objective evaluation in mind. It has the following properties:
- LiveBench is designed to limit potential contamination by releasing new questions monthly, as well as having questions based on recently-released datasets, arXiv papers, news articles, and IMDb movie synopses.
- Each question has verifiable, objective ground-truth answers, allowing hard questions to be scored accurately and automatically, without the use of an LLM judge.
- LiveBench currently contains a set of 18 diverse tasks across 6 categories, and we will release new, harder tasks over time.

This is the instruction_following category of livebench.

See more in our [paper](https://arxiv.org/abs/2406.19314), [leaderboard](https://livebench.ai/), and [datasheet](https://github.com/LiveBench/LiveBench/blob/main/docs/DATASHEET.md).