# Balance 训练数据

基于 v2 的 800 道题，只下采样各训练折中 gold label 为 `gpt-5-nano/low` 的题。原始数据保留，测试折不变。

| 数据集 | 每折原训练题数 | 删除 | 保留 | 测试题数 |
|---|---:|---:|---:|---:|
| LiveBench | 200 | 100 | 100 | 50 |
| SuperGPQA | 240 | 120 | 120 | 60 |
| MMLU-Pro | 200 | 100 | 100 | 50 |
| 合计 | 640 | 320 | 320 | 160 |

## 文件

- `quality/`、`balanced/`、`cost/`：三个奖励设置，α/β 分别为 1/0、0.5/0.5、0.2/0.8。
- 各设置下 `fold_0/` 至 `fold_4/`：每折包含 `train_balance.jsonl`、`test.jsonl`、`manifest.json`。
- `manifest.json`：保留题的 gold label（配置索引）、删除题 ID、原训练折归一化参数、领域配额和文件校验值。
- `sampling_manifest.json`：输入校验值、配置索引顺序和全部划分索引。
- `sample.py`：生成代码；在项目根目录运行 `python3 data/sampled_v2_balance/sample.py`。

## 关键决策

- 每个 setting 独立下采样；每个 setting 五折累计 1,600 次训练样本，不是 1,600 道独立题目。
- 固定原 640 题训练折计算的奖励归一化参数和 gold label，删题后不重新计算。
- 三个 dataset 比例严格保持 31.25% / 37.50% / 31.25%。各 task group 优先减半；nano low 不足的组，将缺额分给同一 dataset 的其他组。
- 随机种子 224；所有模型训练随机种子共用这些划分。题目记录格式与原数据一致。
- 测试集仍为原始每折 160 题。该目录是按 setting 和 fold 划分的数据，不能将训练文件合并后重新随机划分。
- 已完成 balance MLP 训练，运行 `router/mlp.py --balance --workers 4` 使用本目录划分与固定标签；结果见 [实验报告](../../reports/router_reward_comparison.md)，环境与复现见 [训练说明](../../router/TRAINING.md)。不加 `--balance` 时仍使用原始数据。

## 下采样后的 Gold Label 比例

| 配置 | 质量优先 | 质量/成本均衡 | 省钱优先 |
|---|---:|---:|---:|
| GPT-5-nano low | 35.50% | 35.50% | 55.31% |
| GPT-5-nano medium | 13.25% | 13.50% | 12.25% |
| GPT-5-nano high | 4.50% | 4.50% | 3.25% |
| GPT-5-mini low | 27.50% | 27.75% | 25.44% |
| GPT-5-mini medium | 6.00% | 7.00% | 3.00% |
| GPT-5-mini high | 3.75% | 3.25% | 0.25% |
| GPT-5 low | 6.25% | 6.00% | 0.50% |
| GPT-5 medium | 2.00% | 1.75% | 0.00% |
| GPT-5 high | 1.25% | 0.75% | 0.00% |

比例汇总各 setting 的五折训练标签，分母为 1,600。
