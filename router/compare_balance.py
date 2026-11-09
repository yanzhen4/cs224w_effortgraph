"""Compare original and balance routers on identical test folds and reward scales."""
import collections
import json

import numpy as np

import data


def evaluate(table, name):
    records = [json.loads(line) for line in (data.RUNS / 'router' / name / 'picks.jsonl').read_text().splitlines()]
    assert len(records) == 75
    index = {q: i for i, q in enumerate(table.ids)}
    jobs = set()
    result = {}
    for alpha, beta in data.SETTINGS:
        seed_metrics = []
        fold_metrics = collections.defaultdict(list)
        shares = np.zeros(9, dtype=int)
        train_acc = []
        for seed in range(5):
            scores, costs, rewards, datasets = [], [], [], []
            seen = set()
            for r in records:
                if (r['alpha'], r['beta'], r['seed']) != (alpha, beta, seed):
                    continue
                job = (alpha, beta, seed, r['fold'])
                assert job not in jobs
                jobs.add(job)
                ids = r['sample_ids']
                expected = {q for q, f in zip(table.ids, table.fold) if f == r['fold']}
                assert len(ids) == 160 and set(ids) == expected and not seen.intersection(ids)
                seen.update(ids)
                ii = np.array([index[q] for q in ids])
                kk = np.array(r['picks'])
                assert kk.shape == (160,) and ((kk >= 0) & (kk < 9)).all()
                train = table.fold != r['fold']
                norm = data.Normalizer(table.P[train], table.C[train])
                p, c = table.P[ii, kk], table.C[ii, kk]
                reward = norm.reward(p, c, alpha, beta)
                scores.extend(p); costs.extend(c); rewards.extend(reward)
                datasets.extend(table.dataset[ii])
                fold_metrics[r['fold']].append({'score': float(p.mean()), 'cost': float(c.mean()),
                                                'reward': float(reward.mean())})
                shares += np.bincount(kk, minlength=9)
                train_acc.append(r['train_label_accuracy'])
                if name == 'mlp_balance':
                    assert r['train_questions'] == 320 and r['test_questions'] == 160
            assert seen == set(table.ids)
            seed_metrics.append({'seed': seed, 'score': float(np.mean(scores)), 'cost': float(np.mean(costs)),
                                 'reward': float(np.mean(rewards)),
                                 'dataset_scores': {d: float(np.array(scores)[np.array(datasets) == d].mean())
                                                    for d in data.DATASETS}})
        result[f'{alpha},{beta}'] = {
            'mean': {k: float(np.mean([m[k] for m in seed_metrics])) for k in ('score', 'cost', 'reward')},
            'sd': {k: float(np.std([m[k] for m in seed_metrics], ddof=1)) for k in ('score', 'cost', 'reward')},
            'dataset_scores': {d: float(np.mean([m['dataset_scores'][d] for m in seed_metrics])) for d in data.DATASETS},
            'seeds': seed_metrics,
            'folds': {f: {k: float(np.mean([v[k] for v in values])) for k in ('score', 'cost', 'reward')}
                      for f, values in sorted(fold_metrics.items())},
            'pick_share': {f'{m}/{e}': float(shares[k] / shares.sum()) for k, (m, e) in enumerate(data.CONFIGS)},
            'train_label_accuracy': float(np.mean(train_acc))}
    assert len(jobs) == 75
    return result


def main():
    table = data.load()
    results = {name: evaluate(table, name) for name in ('mlp', 'mlp_balance')}
    out = data.RUNS / 'router/mlp_balance/comparison.json'
    out.write_text(json.dumps(results, indent=2) + '\n')
    lines = ['## Balance 训练实验结果', '',
             '保持原超参：1,000 epochs、Adam lr=0.001、batch=32、hidden=32；每个 setting 做 5 折 × 5 seeds。'
             '训练题由每折 640 减到 320，测试题及原训练折的 reward 归一化参数不变。', '',
             '| 设置 | 训练数据 | 平均得分 | 每题成本（USD） | Reward |',
             '|---|---|---:|---:|---:|']
    for (alpha, beta), setting in zip(data.SETTINGS, ('质量优先', '质量/成本均衡', '省钱优先')):
        key = f'{alpha},{beta}'
        for name, label in (('mlp', '原始'), ('mlp_balance', 'balance')):
            r = results[name][key]['mean']
            lines.append(f"| {setting} | {label} | {r['score']:.2%} | ${r['cost']:.6f} | {r['reward']:.4f} |")
    lines += ['', '结果为 5 个随机种子的测试均值。每个种子汇总全部 800 道题的 out-of-fold 预测。', '',
              'Balance 提高了三个设置的答题得分，但也增加了调用成本：质量优先的 reward 提升，均衡和省钱设置的 reward 下降。', '',
              '| 设置 | 原始 nano low 选择占比 | Balance nano low 选择占比 |', '|---|---:|---:|']
    for (a, b), setting in zip(data.SETTINGS, ('质量优先', '质量/成本均衡', '省钱优先')):
        key = f'{a},{b}'
        shares = [results[n][key]['pick_share']['gpt-5-nano/low'] for n in ('mlp', 'mlp_balance')]
        lines.append(f'| {setting} | {shares[0]:.2%} | {shares[1]:.2%} |')
    lines += ['', '复现命令与环境见 [训练说明](../router/TRAINING.md)。'
              '完整指标（含逐折、逐 seed、标准差及分数据集得分）见 [comparison.json](../runs/router/mlp_balance/comparison.json)。', '']
    path = data.ROOT / 'reports/router_reward_comparison.md'
    text = path.read_text()
    text = text.replace('尚未使用 balance 数据重新训练；上方 reward 表仍为原始训练数据的结果。',
                        '上方 reward 表为原始训练数据的结果；balance 训练结果见下文。')
    text = text.split('## Balance 训练实验结果')[0].rstrip()
    path.write_text(text + '\n\n' + '\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
