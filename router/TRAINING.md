# MLP 训练

本机独立环境：`/Users/yanzhenshen/Desktop/CS224W/.venvs/effortgraph`。依赖缓存和运行缓存均放在 `CS224W/.cache/`；不修改系统 Python。当前实验使用预计算 MiniLM embedding，无需下载或微调 MiniLM。

## 环境与复现

在 `EffortGraph` 根目录执行：

```bash
export UV_CACHE_DIR="$PWD/../.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/../.python"
export HF_HOME="$PWD/../.cache/huggingface"
export TORCH_HOME="$PWD/../.cache/torch"
export XDG_CACHE_HOME="$PWD/../.cache"
export PYTHONPYCACHEPREFIX="$PWD/../.cache/pycache"
python3 -m venv ../.venvs/effortgraph
uv pip install --python ../.venvs/effortgraph/bin/python -r router/requirements-cpu.lock.txt
../.venvs/effortgraph/bin/python router/mlp.py --balance --workers 4
../.venvs/effortgraph/bin/python router/compare_balance.py
```

依赖版本见 `requirements-cpu.lock.txt`。每个训练进程使用一个 CPU 线程。

## Balance 实验

- 数据：`data/sampled_v2_balance/{quality,balanced,cost}/fold_{0..4}/`。
- 固定读取各折的 320 道训练题和 gold label；测试题仍为原来的 160 道。使用原 640 题训练折的归一化参数评估 reward。
- 超参沿用原实验：31,105 个可训练参数，Adam lr=0.001，batch=32，1,000 epochs，BCE loss，无 dropout、weight decay、验证集或早停，使用最后一轮模型。
- 三种奖励设置 × 五折 × 五个种子（0–4），共 75 次训练。每次 10,000 次更新；原实验每次 20,000 次更新，epoch 数相同。
- `runs/router/mlp_balance/config.json`：训练配置；`picks.jsonl`：测试题选择；`checkpoints/`：75 个最终模型权重。
- `runs/router/mlp_balance/comparison.json`：新旧结果、逐折及逐 seed 指标；`reports/router_reward_comparison.md`：对比表。
- 原始 `runs/router/mlp/` 不覆盖。重新运行同名实验会覆盖该实验的结果文件。

不加 `--balance` 时仍使用原始训练划分。可用 `--output` 指定新实验目录；`--epochs`、`--fold`、`--setting`、`--seeds` 可用于小规模检查。

## MiniLM 全参数微调：单折试验

`finetune_full.py` 使用 quality、fold 0、seed 0 的 balance 数据（320 train / 160 test），从该折已训练好的 MLP 权重继续训练；MiniLM 从预训练权重开始，全部编码器参数解冻。均值池化与单位长度归一化和原 embedding 一致，不使用 BERT pooler。

- 固定 5 epochs，最后一轮评估，不按测试成绩选择模型。
- CPU 4 线程，microbatch=8，累积 4 次梯度，有效 batch=32。
- AdamW：MiniLM lr=2e-5，MLP lr=1e-4，weight decay=0.01，gradient clipping=1。
- 分词结果缓存于 `CS224W/.cache/effortgraph/`；模型下载缓存仍在 `CS224W/.cache/huggingface/`。
- 输出：`runs/router/full_balance_quality_fold0_seed0/`，含配置、逐轮耗时、结果、完整模型权重。

使用上文相同环境变量，安装 `router/requirements-full-cpu.lock.txt` 后运行：

```bash
../.venvs/effortgraph/bin/python router/finetune_full.py --epochs 5 --threads 4
```

需要提前将 `sentence-transformers/all-MiniLM-L6-v2` 下载到上述缓存；训练仅从本地加载。单折结果与同一折、同一种子的冻结编码器基线比较，不等同于完整五折五种子结果。

## 两种初始化：50 epochs，单 seed 完整五折

```bash
../.venvs/effortgraph/bin/python -u router/run_full_suite.py --workers 2 --threads 2
```

- Warm-start 加载对应的 balance MLP；scratch 随机初始化 MLP；两者 MiniLM 均从同一预训练权重开始，之后编码器和 MLP 一起更新。
- 每种方案 3 settings × 5 folds × seed 0，共 30 次训练。每次 50 epochs、500 次更新，微调超参同上；每 10 轮保存模型权重，不保存优化器状态。
- 输出目录：`runs/router/full_balance_50ep_seed0/`。`status.json` 记录任务状态，`runner.log` 记录整体进度；每个方案/setting/fold 子目录包含 `training.log`、`history.json`、5 个 checkpoint 和最终 `result.json`。
- 总表只采用第 50 轮的结果，不按测试成绩选择 checkpoint。完成后自动生成 `summary.json` 并更新 `reports/router_reward_comparison.md`，同时列出 seed=0 的冻结编码器基线。
- 重新运行会跳过已经完成且保存齐全的任务；未完成的任务从该次实验的初始权重重跑，不从中间权重恢复优化器。
