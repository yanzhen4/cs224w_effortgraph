# 数据统计与关键决策

Balance 实验版：基于 v2 800 题，每折仅删除 nano low 标签的 320 道训练题，保留 LiveBench / SuperGPQA / MMLU-Pro 各 100 / 120 / 100 题；测试集不变。详细比例与决策见 [Balance 数据说明](sampled_v2_balance/README.md)。下文为原始 1,800 题版本的统计。

## 总体规模

| 数据集 | 原始记录数 | 抽样题数 | Task groups |
|---|---:|---:|---:|
| LiveBench | 1,576 | 600 | 6 |
| SuperGPQA | 26,529 | 600 | 6 |
| MMLU-Pro | 12,102 | 600 | 5 |
| 合计 | 40,207 | 1,800 | 17 |

- 使用已下载的固定版本，随机种子统一为 `224`。
- Task group 按数据集与类别/领域区分。
- 抽样前去除池内完全相同的题目文本，采用无放回抽样。
- 尚未划分 train / validation / test。

## LiveBench

原始 test 共 1,436 道，另有 Language train 140 道。仅从 test 抽样。

| 类别 | 原始 test | 抽样 |
|---|---:|---:|
| Coding | 128 | 100 |
| Data Analysis | 150 | 100 |
| Instruction Following | 400 | 100 |
| Mathematics | 368 | 100 |
| Reasoning | 200 | 100 |
| Language | 190 | 100 |

关键决策：

- 六类各 100 道，先按子任务均衡分配，再覆盖已有难度标签。
- 保留历史移除题及其日期标记；Coding 原始题均标记于 2025-04-02 移除。
- 有原生难度则保留，无难度则不额外添加 `unknown`。
- Coding 样本为 easy 29、medium 46、hard 25；Math 有 33 道带 hardness，Reasoning 有 31 道带 level。

## SuperGPQA

完整题库包含 72 个 field；难度分布为 easy 7,731、middle 11,748、hard 7,050。

| 选定领域 | 原始总数 | 抽样 middle | 抽样 hard |
|---|---:|---:|---:|
| Physics | 2,845 | 50 | 50 |
| Mathematics | 2,622 | 50 | 50 |
| Chemistry | 1,769 | 50 | 50 |
| Clinical Medicine | 1,218 | 50 | 50 |
| Biology | 1,120 | 50 | 50 |
| Mechanics | 908 | 50 | 50 |

关键决策：六个领域各 100 道，middle 与 hard 各半，不抽 easy。

完整领域统计见 `raw/supergpqa/field_counts.json`。

## MMLU-Pro

完整数据包含 test 12,032 道、validation 70 道，共 14 类。仅从 test 抽样。

| 选定类别 | 原始 test | 抽样 |
|---|---:|---:|
| Physics | 1,299 | 200 |
| Math | 1,351 | 100 |
| Chemistry | 1,132 | 100 |
| Health | 818 | 100 |
| Biology | 717 | 100 |

关键决策：

- Physics 抽 200 道，其余四类各 100 道。
- 类别内按 `src` 来源标签均衡抽样，共覆盖 41 个来源。
- 使用五个类别作为 Task groups，不单独设 Mechanics。
- 无统一难度标签，不做难度分层。

各数据集的详细抽样数量与来源见 `sampled/{dataset}/sampling_manifest.json`。
