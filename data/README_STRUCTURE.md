# 文件结构

```text
data/
├── raw/                         # 完整原始数据
│   ├── livebench/
│   ├── supergpqa/
│   ├── mmlu_pro/
│   ├── download_datasets.py     # 下载与 JSONL 转换代码
│   ├── manifest.json            # 数据版本与文件清单
│   └── inventory.json           # 原始数据统计
├── sampled/
│   ├── livebench/
│   ├── supergpqa/
│   └── mmlu_pro/
│       # 每个目录包含：
│       # all.jsonl              # 600 道样本
│       # <类别或领域>.jsonl      # 分类样本
│       # sample.py              # 抽样代码
│       # sampling_manifest.json # 抽样明细
├── combined_1800.jsonl          # 三个数据集合并后的 1,800 道题
├── build_combined.py            # 合并代码
├── README_STRUCTURE.md         # 文件结构与使用方式
└── README_STATISTICS.md         # 数据统计与关键决策
```

`raw` 中的 `jsonl/` 保存完整原始数据的 JSONL 转换。

## 使用

全部实验题目读取 `combined_1800.jsonl`；单个数据集读取 `sampled/{dataset}/all.jsonl`。

合并文件每行包含：

| 字段 | 含义 |
|---|---|
| `sample_id` | 带数据集前缀的唯一 ID |
| `dataset` | 数据集名称 |
| `source_id` | 原始题目 ID |
| `task_group` | 数据集 + 类别/领域 |
| `record` | 完整原始题目记录 |

## 复现

在项目根目录运行：

```bash
python3 data/sampled/livebench/sample.py
python3 data/sampled/supergpqa/sample.py
python3 data/sampled/mmlu_pro/sample.py
python3 data/build_combined.py
```

抽样与合并使用 Python 标准库，随机种子均为 `224`。下载代码为 `data/raw/download_datasets.py`，需要 curl 和 pyarrow。
