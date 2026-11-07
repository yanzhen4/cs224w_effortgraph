"""Markdown version of the EffortGraph baseline report, from the same run outputs as build_report.py."""
import collections
from pathlib import Path

import numpy as np

import build_report as b  # builds the HTML page too; reuses its loaded data and derived numbers

OUT = b.REPO / 'reports/baseline_report.md'
data, S, fixed, LABEL, NAMES, ctx = b.data, b.S, b.fixed, b.LABEL, b.NAMES, b.ctx


def table(head, rows, align=None):
    align = align or ['l'] + ['r'] * (len(head) - 1)
    sep = ['---:' if a == 'r' else ':---' for a in align]
    lines = ['| ' + ' | '.join(head) + ' |', '| ' + ' | '.join(sep) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(lines)


money = b.money
DS = data.DATASETS

fixed_rows = [[LABEL[n], *(f"{fixed[n][d]:.3f}" for d in DS), f"**{fixed[n]['all']:.3f}**", money(fixed[n]['cost']),
               f"${b.tot_cost[n]:.2f}"] for n in NAMES]
fixed_rows.append(['随机（9 个配置的期望）', *(f"{fixed['random'][d]:.3f}" for d in DS),
                   f"**{fixed['random']['all']:.3f}**", money(fixed['random']['cost']), '—'])

group_rows = []
for g in b.groups:
    vals = [np.mean(b.cell[(g, m, e)]) for m, e in data.CONFIGS]
    ds, name = g.split('/', 1)
    group_rows.append([f"{b.DS_NAME[ds]} · {name.replace('_', ' ')}"] +
                      [f"**{v:.2f}**" if v == max(vals) else f"{v:.2f}" for v in vals])

route_rows = []
for k, v in S.items():
    m, sd = v['routers']['mlp']['mean'], v['routers']['mlp']['sd']
    name = f"{b.SETTING_NAME[k]} {b.SETTING_AB[k]}"
    route_rows.append([name, 'MLP 路由（5 个种子）', *(f"{m[d]:.3f} ± {sd[d]:.3f}" for d in DS),
                       f"{m['macro']:.3f} ± {sd['macro']:.3f}", f"**{m['all']:.3f}** ± {sd['all']:.3f}", money(m['cost'], 5)])
    for ref, nm in (('per-group best', '按题组选'), ('oracle', '理想路由')):
        r = v['references'][ref]
        route_rows.append(['', nm, *(f"{r[d]:.3f}" for d in DS), f"{r['macro']:.3f}", f"**{r['all']:.3f}**", money(r['cost'], 5)])

diff_rows = []
for k, v in S.items():
    vs = v['routers']['mlp']['vs']
    p, g = vs['per-group best'], vs['gpt-5/high']
    diff_rows.append([b.SETTING_NAME[k], f"{p[0]:+.3f}（{p[1]:+.3f} 到 {p[2]:+.3f}）", f"{g[0]:+.3f}（{g[1]:+.3f} 到 {g[2]:+.3f}）"])

pick_rows = [[LABEL[n]] + [f"{b.LABEL_SHARE[k].get(n, 0):.0%} / {S[k]['routers']['mlp']['pick_share'].get(n, 0):.0%}" for k in S]
             for n in NAMES]

strata_rows = [[r[0], r[1], r[2], r[3]] for r in b.strata_rows]

c = ctx
md = f"""# EffortGraph 基线结果

三个 GPT-5 模型各开三个推理档位，共 9 个配置，在 800 道题上各答一次。再用这些结果训练、评估一个不用图的 MLP 路由器，作为之后 GNN 路由器的对照。

800 道题 · 15 个题组 · 9 个配置 · {c['n_calls']} 次调用 · 按标价 ${c['total_cost']} · 2026-11-07

## 要点

- **换大模型比开高档更管用。** nano 换成 gpt-5（都用 low），得分高 {c['size_gain'].lstrip('+')}；gpt-5 从 low 开到 high，只高 {c['effort_gain'].lstrip('+')}。每个模型从 low 开到 high，得分只涨 0.01 到 0.08。
- **按题挑配置，上限很高。** 理想路由得 {c['oracle_score']}，每题 {c['oracle_cost']}；固定用 gpt-5 high 得 {c['g5h_score']}，每题 {c['g5h_cost']}。理想路由事后看答案，每道题挑能拿最高分的配置里最便宜的一个，是任何路由器的上限。质量优先时，它比 gpt-5 high 高 0.10 分，花费只有后者的 1/{c['cost_ratio']}。
- **MLP 路由器不如固定用 mini low。** 质量优先时 MLP 得 {c['mlp_score']}，每题 {c['mlp_cost']}；固定用 mini low 得 {c['mini_score']}，每题 {c['mini_cost']}。省钱优先时，MLP 比 nano low 高 0.03，花费是两倍。按题组选（每个题组固定用训练集上收益最高的配置）在质量优先时得 0.773。
- **MLP 多数时候只选 nano low。** 训练标签里 {c['lab_nano']} 是 nano low，测试时 MLP 在 {c['mlp_nano']} 的题上选 nano low。训练标签只把每题收益最高的配置记为正例；质量优先时，它就是答对的配置里最便宜的那个，所以大多是 nano low。MLP 在训练集上几乎记住了全部标签（{c['acc']}），测试时基本照这个比例去选：三种设置下有 {c['mlp_nano_lo']} 到 {c['mlp_nano_hi']} 的题选了 nano low。

## 模型与调用设置

{table(['模型', '版本', '推理档位', '输入', '缓存输入', '输出'], [
    ['gpt-5-nano', '2025-08-07', 'low / medium / high', '$0.05', '$0.005', '$0.40'],
    ['gpt-5-mini', '2025-08-07', 'low / medium / high', '$0.25', '$0.025', '$2.00'],
    ['gpt-5', '2025-08-07', 'low / medium / high', '$1.25', '$0.125', '$10.00']], ['l', 'l', 'l', 'r', 'r', 'r'])}

价格单位是每百万 token 的美元标价。

- 9 个配置是 3 个模型和 low、medium、high 三个推理档位的组合。每道题在每个配置下只答一次。
- 通过微软内部的 TRAPI 接口调用（OpenAI Responses API）。输出上限 64,000 token，推理 token 也计在内。
- 零样本作答，提示词按各基准官方评测代码的方式构造。MMLU-Pro 去掉了官方的 5 个示例，SuperGPQA 用官方零样本模板，LiveBench 直接发原题。
- 花费按标价计算，推理 token 按输出价格计。
- 提案原定用 gpt-5.4 系列。它在这个接口上每分钟只有 1 万 token 的额度，其余模型是 100 万，跑不动这个规模，所以换成同一代的三个 GPT-5 模型。

## 数据集与抽样

{table(['数据集', '每组', '合计', '题组'], [
    ['MMLU-Pro', '50', '250', 'math、physics、health、law、history'],
    ['SuperGPQA', '60', '300', 'Physics、Mathematics、Chemistry、Clinical Medicine、Biology'],
    ['LiveBench', '50', '250', 'math、reasoning、coding、data analysis、instruction following']], ['l', 'r', 'r', 'l'])}

- 每个题组先去重，LiveBench 里提示词完全相同的题只留最新发布的一道。然后用固定的随机种子把题目排成分层轮转的顺序，取前 50 道（SuperGPQA 取 60 道）。顺序固定，任意前缀都是分层随机样本，以后要扩充只需接着往后取。
- 分层方式：MMLU-Pro 按题目来源轮流取，SuperGPQA 按 easy、middle、hard 轮流取，LiveBench 先按子任务、再按官方难度轮流取。
- LiveBench 里已从官方在线题库下架的旧题也保留，250 道里有 127 道，它们都有标准答案。
- 5 折交叉验证按题目在本组顺序里的位置取余 5 来分，每折 160 道，每个题组每折 10 道（SuperGPQA 12 道）。

### 各题组的抽取情况

{table(['题组', '去重后可选', '抽取', '组内分层（抽到的题数）'], strata_rows, ['l', 'r', 'r', 'l'])}

## 评分方式

- MMLU-Pro 用官方的答案抽取规则。抽不出答案记 0 分；官方代码这时会随机猜一个选项，这里不猜。
- SuperGPQA 用官方评测代码的选项抽取规则。
- LiveBench 用它自己的评分代码。代码题实际运行测试用例；指令遵循题用官方检查器；AMPS 数学题在符号比对不能确认时，按官方做法请 gpt-5-mini 判断两个表达式是否等价。

## 固定配置的结果

{table(['配置', 'MMLU-Pro（250 题）', 'SuperGPQA（300 题）', 'LiveBench（250 题）', '全部 800 题', '每题花费', '800 题总花费'], fixed_rows)}

### 各题组的平均得分

每格 50 或 60 道题，每行最高的加粗。

{table(['题组', *(LABEL[n] for n in NAMES)], group_rows)}

## 路由实验的设置

- **收益**：r = α·p − β·c（提案式 3），p 是得分，c 是花费。两者都在训练折上做 min-max 归一化。花费先取对数再归一化，因为原始花费跨了四个数量级，最便宜和最贵的回答相差 2.6 万倍。
- **三种设置**：质量优先 (α, β) = (1, 0)；均衡 (0.5, 0.5)；省钱优先 (0.2, 0.8)。
- **标签**：每道训练题收益最高的配置记为正例，其余 8 个为负例。收益相同时，取这道题上花费更低的配置。
- **输入**：题目向量和题组向量都用 all-MiniLM-L6-v2 计算，384 维。题目向量只用题目内容（选择题是题干加选项，LiveBench 是原题），不含作答说明，超过 256 个 token 的部分截掉。题组向量用题组名称，例如“SuperGPQA: Clinical Medicine”。
- **MLP 结构**：题目向量和题组向量各经过两层（32 维）。模型和推理档位各学一个 32 维向量。四者拼接后再经过两层，给 9 个配置各打一个分，选分最高的那个。
- **训练**：二元交叉熵，Adam，学习率 0.001，每批 32 道题，1,000 个 epoch，用最后一个模型，不设验证集，与提案一致。
- **评估**：5 折交叉验证。每折用其余 4 折的 640 道题训练，在本折 160 道题上测试，所以每道题正好测一次。每折跑 5 个随机种子，三种设置共 75 次训练。报告 800 道题汇总的得分和每题花费，± 是种子之间的标准差。和参照的差值用题目层面的配对重抽样（1 万次）给出 95% 区间。
- **参照**：固定配置；随机，即 9 个配置的期望；按题组选，即每个题组固定用训练折上平均收益最高的配置，不需要训练；理想路由，即每道测试题按实际结果选收益最高的配置，是上限。

## 路由实验的结果

{table(['设置', '方法', 'MMLU-Pro', 'SuperGPQA', 'LiveBench', '三个数据集平均', '全部 800 题', '每题花费'], route_rows, ['l', 'l'] + ['r'] * 6)}

### MLP 与参照的得分差

800 道题，括号内为 95% 区间。

{table(['设置', 'MLP − 按题组选', 'MLP − 固定用 gpt-5 high'], diff_rows)}

### 各配置被选中的比例

每格是“训练标签 / MLP 测试时的选择”。

{table(['配置', '质量优先', '均衡', '省钱优先'], pick_rows)}

路由开销：算题目向量每题约 {c['emb_ms']} 毫秒（CPU），MLP 打分每题约 {c['router_ms']} 毫秒。

## 采集记录

- {c['n_calls']} 次调用全部完成。输入约 {c['tok_in']} 万 token，输出约 {c['tok_out']} 万 token，其中推理约 {c['tok_reason']} 万。按标价共 ${c['total_cost']}。
- 接口网关会切断超过约 4 分钟的单次请求，碰到的基本是 high 档这类推理很长的调用。共有 32 次调用碰到过：27 次重试后在 4 分钟内完成，另外 5 次改为后台提交、轮询取结果。
- {c['trunc']} 个回答用完了 64,000 个输出 token，没有给出答案，记 0 分。
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(md)
print('wrote', OUT, len(md.splitlines()), 'lines')
