"""Run both full-tuning initializations, 3 settings x 5 folds x seed 0, then update the report."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/router/full_balance_50ep_seed0'
SETTINGS = {'quality': (1.0, 0.0), 'balanced': (0.5, 0.5), 'cost': (0.2, 0.8)}


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def aggregate():
    import numpy as np
    import data
    table = data.load()
    index = {q: i for i, q in enumerate(table.ids)}
    summary = {}
    for method in ('warm', 'scratch'):
        summary[method] = {}
        for setting, (alpha, beta) in SETTINGS.items():
            rewards, scores, costs, old_rewards, old_scores, old_costs = [], [], [], [], [], []
            seen = set()
            for fold in range(5):
                folder = OUT / method / setting / f'fold_{fold}'
                r = json.loads((folder / 'result.json').read_text())
                cfg = json.loads((folder / 'config.json').read_text())
                assert (cfg['init'], cfg['epochs'], cfg['seed'], cfg['fold'], cfg['setting']) == (method, 50, 0, fold, setting)
                assert cfg['train_questions'] == 320 and cfg['test_questions'] == 160
                assert len(list(folder.glob('checkpoint_epoch_*.pt'))) == 5
                ids = r['sample_ids']
                assert len(ids) == len(set(ids)) == 160 and not seen.intersection(ids)
                assert set(ids) == {q for q, f in zip(table.ids, table.fold) if f == fold}
                seen.update(ids)
                ii = [index[q] for q in ids]
                picks = np.array(r['picks'])
                assert picks.shape == (160,) and ((picks >= 0) & (picks < 9)).all()
                p, c = table.P[ii, picks], table.C[ii, picks]
                train = table.fold != fold
                norm = data.Normalizer(table.P[train], table.C[train])
                reward = norm.reward(p, c, alpha, beta)
                assert abs(reward.mean() - r['full_finetune']['reward']) < 1e-10
                rewards.extend(reward); scores.extend(p); costs.extend(c)
                old_rewards.append(r['baseline']['reward']); old_scores.append(r['baseline']['score']); old_costs.append(r['baseline']['cost'])
            assert seen == set(table.ids)
            summary[method][setting] = {'reward': float(np.mean(rewards)), 'score': float(np.mean(scores)),
                                       'cost': float(np.mean(costs)),
                                       'baseline': {'reward': float(np.mean(old_rewards)), 'score': float(np.mean(old_scores)),
                                                    'cost': float(np.mean(old_costs))}}
    write_json(OUT / 'summary.json', summary)
    path = ROOT / 'reports/router_reward_comparison.md'
    text = path.read_text()
    labels = [('MLP router（balance，seed=0）', [summary['warm'][s]['baseline']['reward'] for s in SETTINGS]),
              ('全参数 Warm-start（balance，50 epochs，seed=0）', [summary['warm'][s]['reward'] for s in SETTINGS]),
              ('全参数从头联合训练（balance，50 epochs，seed=0）', [summary['scratch'][s]['reward'] for s in SETTINGS])]
    lines = text.splitlines()
    lines = [line for line in lines if not any(line.startswith('| ' + label + ' |') for label, _ in labels)]
    pos = next(i for i, line in enumerate(lines) if line.startswith('| Oracle（'))
    lines[pos:pos] = ['| ' + label + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |' for label, values in labels]
    # Recompute the non-oracle star marks after adding new strategies.
    import re
    start = next(i for i, line in enumerate(lines) if line.startswith('| 策略 |')) + 2
    end = next(i for i in range(start, len(lines)) if not lines[i].startswith('|'))
    table_rows = []
    for i in range(start, end):
        cells = [c.strip() for c in lines[i].strip('|').split('|')]
        values = [float(re.sub(r'[★*\s]', '', c)) for c in cells[1:]]
        table_rows.append((i, cells[0], values))
    best = [max(v[k] for _, name, v in table_rows if not name.startswith('Oracle')) for k in range(3)]
    for i, name, values in table_rows:
        cells = [f'**{v:.4f}**' if name.startswith('Oracle') else
                 (f'**{v:.4f} ★**' if v == best[k] else f'{v:.4f}') for k, v in enumerate(values)]
        lines[i] = '| ' + name + ' | ' + ' | '.join(cells) + ' |'
    text = '\n'.join(lines).rstrip()
    marker = '## 全参数微调：50 epochs，seed=0'
    text = text.split(marker)[0].rstrip()
    extra = [marker, '', '两种初始化各运行 3 settings × 5 folds × 1 seed。每折 320 题训练、160 题测试；表中为全部 800 道题的 out-of-fold 平均。仅使用第 50 轮结果，不按测试集挑选 checkpoint。原始五种子结果保留，另列相同 seed=0 的冻结编码器基线供直接比较。', '',
             '| 设置 | 初始化 | 平均得分 | 每题成本 | Reward |', '|---|---|---:|---:|---:|']
    for setting in SETTINGS:
        for method in ('warm', 'scratch'):
            r = summary[method][setting]
            extra.append(f"| {setting} | {method} | {r['score']:.2%} | ${r['cost']:.6f} | {r['reward']:.4f} |")
    extra += ['', 'Warm-start 从对应的已训练 balance MLP 开始；scratch 的 MLP 随机初始化。两者 MiniLM 均从同一预训练权重开始并全部解冻。优化器与学习率相同：AdamW，encoder=2e-5，MLP=1e-4，weight decay=0.01，有效 batch=32。每 10 轮保存权重，不保存优化器状态。', '',
              '完整结果与运行记录：[实验目录](../runs/router/full_balance_50ep_seed0/)。', '']
    path.write_text(text + '\n\n' + '\n'.join(extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # Only one suite runner may write this output directory.
    import fcntl
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [(method, setting, fold) for fold in range(5) for setting in SETTINGS for method in ('warm', 'scratch')]
    started = time.time()
    status = {'pid': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'workers': args.workers, 'threads_per_job': args.threads, 'epochs': 50,
              'seed': 0, 'jobs_total': 30, 'completed': [], 'failed': [], 'state': 'running',
              'code_sha256': hashlib.sha256((ROOT / 'router/finetune_full.py').read_bytes()).hexdigest()}
    write_json(OUT / 'status.json', status)
    env = os.environ.copy()
    cache = ROOT.parent / '.cache'
    env.update(HF_HOME=str(cache/'huggingface'), TORCH_HOME=str(cache/'torch'), XDG_CACHE_HOME=str(cache),
               PYTHONPYCACHEPREFIX=str(cache/'pycache'), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

    def run(job):
        method, setting, fold = job
        dest = OUT / method / setting / f'fold_{fold}'
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'result.json').exists():
            cfg = json.loads((dest / 'config.json').read_text())
            assert (cfg['epochs'], cfg['init'], cfg['setting'], cfg['fold'], cfg['seed']) == (50, method, setting, fold, 0)
            assert all((dest/f'checkpoint_epoch_{e:03d}.pt').exists() for e in (10, 20, 30, 40, 50))
            return job, 0, 'already complete'
        cmd = [sys.executable, '-u', str(ROOT/'router/finetune_full.py'), '--epochs', '50', '--save-every', '10',
               '--init', method, '--setting', setting, '--fold', str(fold), '--seed', '0', '--threads', str(args.threads),
               '--output', str(dest)]
        with (dest/'training.log').open('w') as log:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(dest/'process.json', {'pid': p.pid, 'command': cmd, 'started': time.time()})
            code = p.wait()
        return job, code, 'finished'

    print('Starting 30 jobs: warm + scratch, 3 settings, 5 folds, seed 0, 50 epochs.', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                job, code, message = future.result()
            except Exception as exc:
                job, code, message = futures[future], -1, repr(exc)
            status['completed' if code == 0 else 'failed'].append({'job': job, 'returncode': code, 'message': message})
            status['elapsed_s'] = time.time() - started
            write_json(OUT/'status.json', status)
            print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {len(status['completed'])}/30 complete, {len(status['failed'])} failed: {job}", flush=True)
    if not status['failed']:
        try:
            aggregate()
            status['state'] = 'complete'
        except Exception as exc:
            status['state'] = 'report_failed'
            status['report_error'] = repr(exc)
            raise
        finally:
            status['elapsed_s'] = time.time() - started
            write_json(OUT/'status.json', status)
    else:
        status['state'] = 'failed'
        write_json(OUT/'status.json', status)
        raise SystemExit('Some jobs failed; inspect training.log and rerun to retry unfinished jobs.')


if __name__ == '__main__':
    main()
