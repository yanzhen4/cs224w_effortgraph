---
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: category
    dtype: string
  - name: ground_truth
    dtype: string
  - name: turns
    sequence: string
  - name: group
    dtype: string
  - name: movie_name
    dtype: string
  - name: release_date
    dtype: string
  - name: task
    dtype: string
  - name: livebench_release_date
    dtype: timestamp[s]
  - name: livebench_removal_date
    dtype: string
  - name: citation
    dtype: string
  - name: raw_id
    dtype: int64
  splits:
  - name: test
    num_bytes: 534070
    num_examples: 190
  download_size: 287876
  dataset_size: 534070
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
arxiv: 2406.19314
---
# Dataset Card for "livebench/language"
LiveBench is a benchmark for LLMs designed with test set contamination and objective evaluation in mind. It has the following properties:
- LiveBench is designed to limit potential contamination by releasing new questions monthly, as well as having questions based on recently-released datasets, arXiv papers, news articles, and IMDb movie synopses.
- Each question has verifiable, objective ground-truth answers, allowing hard questions to be scored accurately and automatically, without the use of an LLM judge.
- LiveBench currently contains a set of 18 diverse tasks across 6 categories, and we will release new, harder tasks over time.
This is the instruction_following category of livebench.

See more in our [paper](https://arxiv.org/abs/2406.19314), [leaderboard](https://livebench.ai/), and [datasheet](https://github.com/LiveBench/LiveBench/blob/main/docs/DATASHEET.md).
