# EffortGraph

CS224W final project: joint model and reasoning effort selection with graph neural networks.

- Three datasets: LiveBench, SuperGPQA, and MMLU-Pro.
- 600 sampled questions per dataset; 1,800 questions across 17 task groups.
- [Data structure and reproduction](data/README_STRUCTURE.md)
- [Dataset statistics and sampling decisions](data/README_STATISTICS.md)
- [GPT-5 baseline report](reports/baseline_report.md): 800-question v2 set (`data/sampled_v2`), 9 configurations (gpt-5-nano / gpt-5-mini / gpt-5 x low / medium / high), responses and scores in `runs/`, MLP router baseline.
- [Balance router comparison](reports/router_reward_comparison.md): original versus balanced training data; [CPU training and environment](router/TRAINING.md).

## Current results

The current pilot contains 800 questions and 7,200 scored model responses. The table compares seed 0 across five folds, using identical held-out questions and the final checkpoint. Balance uses 320 training questions per fold; Original MLP uses 640. Full fine-tuning jointly updates MiniLM and the routing head for 50 epochs.

| Router | Quality-first | Balanced | Cost-first |
|---|---:|---:|---:|
| Original MLP | 0.6741 | 0.1835 | -0.0865 |
| Balance MLP | 0.6923 | 0.1708 | -0.1266 |
| Full fine-tuning, warm start | 0.6916 | 0.1801 | -0.1088 |
| Full fine-tuning, fresh head | 0.6005 | 0.1879 | -0.0597 |

Increasing trainable parameters from 31,105 to 22.60M did not yield consistent reward improvements. Warm start improves the cost-aware Balance MLP results but remains below Original MLP; fresh-head training always selects nano low. No learned router beats the best fixed configuration for its objective. The graph router is the next experiment.

- [Milestone report (PDF)](reports/milestone/milestone.pdf) · [LaTeX source](reports/milestone/milestone.tex), using the proposal's NeurIPS template.
- Training: [MLP](router/mlp.py), [full fine-tuning](router/finetune_full.py), [suite runner](router/run_full_suite.py).
- [Balance splits and sampling code](data/sampled_v2_balance/); predictions and metrics in `runs/router/`.
- Recompute the milestone table: `python reports/milestone/build_metrics.py`. Model checkpoints remain local.

## Download

Install [Git LFS](https://git-lfs.com/), then run:

```bash
git lfs install
git clone https://github.com/yanzhen4/cs224w_effortgraph.git
cd cs224w_effortgraph
git lfs pull
```

JSONL and Parquet files are stored with Git LFS. Sampling scripts are in `data/sampled/{dataset}/sample.py`.
