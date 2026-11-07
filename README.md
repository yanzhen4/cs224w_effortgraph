# EffortGraph

CS224W final project: joint model and reasoning effort selection with graph neural networks.

- Three datasets: LiveBench, SuperGPQA, and MMLU-Pro.
- 600 sampled questions per dataset; 1,800 questions across 17 task groups.
- [Data structure and reproduction](data/README_STRUCTURE.md)
- [Dataset statistics and sampling decisions](data/README_STATISTICS.md)
- [GPT-5 baseline report](reports/baseline_report.md): 800-question v2 set (`data/sampled_v2`), 9 configurations (gpt-5-nano / gpt-5-mini / gpt-5 x low / medium / high), responses and scores in `runs/`, MLP router baseline.

## Download

Install [Git LFS](https://git-lfs.com/), then run:

```bash
git lfs install
git clone https://github.com/yanzhen4/cs224w_effortgraph.git
cd cs224w_effortgraph
git lfs pull
```

JSONL and Parquet files are stored with Git LFS. Sampling scripts are in `data/sampled/{dataset}/sample.py`.
