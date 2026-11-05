---
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: category
    dtype: string
  - name: turns
    sequence: string
  - name: question_title
    dtype: string
  - name: public_test_cases
    dtype: string
  - name: private_test_cases
    dtype: string
  - name: original_json
    struct:
    - name: question_title
      dtype: string
    - name: question_content
      dtype: string
    - name: platform
      dtype: string
    - name: question_id
      dtype: string
    - name: contest_id
      dtype: string
    - name: contest_date
      dtype: timestamp[s]
    - name: starter_code
      dtype: string
    - name: difficulty
      dtype: string
    - name: metadata
      dtype: string
  - name: release_date
    dtype: timestamp[s]
  - name: citation
    dtype: string
  - name: task
    dtype: string
  - name: livebench_release_date
    dtype: timestamp[s]
  - name: livebench_removal_date
    dtype: timestamp[s]
  - name: remainder
    dtype: string
  - name: solution
    dtype: string
  - name: partial_solution
    dtype: string
  splits:
  - name: test
    num_bytes: 254934173
    num_examples: 128
  download_size: 244785858
  dataset_size: 254934173
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
arxiv: 2406.19314
---
# Dataset Card for "livebench/coding"
LiveBench is a benchmark for LLMs designed with test set contamination and objective evaluation in mind. It has the following properties:
- LiveBench is designed to limit potential contamination by releasing new questions monthly, as well as having questions based on recently-released datasets, arXiv papers, news articles, and IMDb movie synopses.
- Each question has verifiable, objective ground-truth answers, allowing hard questions to be scored accurately and automatically, without the use of an LLM judge.
- LiveBench currently contains a set of 18 diverse tasks across 6 categories, and we will release new, harder tasks over time.
This is the instruction_following category of livebench.

See more in our [paper](https://arxiv.org/abs/2406.19314), [leaderboard](https://livebench.ai/), and [datasheet](https://github.com/LiveBench/LiveBench/blob/main/docs/DATASHEET.md).
