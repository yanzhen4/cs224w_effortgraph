"""Build the EffortGraph baseline report page from the run outputs (scores, routing report, sampling manifest)."""
import collections
import html
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'router'))
import data  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).with_name('effortgraph_baseline.html')
t = data.load()
rep = json.load(open(data.RUNS / 'router/report.json'))
manifest = json.load(open(REPO / 'data/sampled_v2/sampling_manifest.json'))
scores = [json.loads(line) for line in open(data.RUNS / 'full/scores.jsonl')]
picks_jobs = [json.loads(line) for line in open(data.RUNS / 'router/mlp/picks.jsonl')]

SHORT = {'gpt-5-nano': 'nano', 'gpt-5-mini': 'mini', 'gpt-5': 'gpt-5'}
NAMES = [f"{m}/{e}" for m, e in data.CONFIGS]
LABEL = {f"{m}/{e}": f"{SHORT[m]} {e}" for m, e in data.CONFIGS}
SETTING_NAME = {'alpha=1.0,beta=0.0': '质量优先', 'alpha=0.5,beta=0.5': '均衡', 'alpha=0.2,beta=0.8': '省钱优先'}
SETTING_AB = {'alpha=1.0,beta=0.0': '(1, 0)', 'alpha=0.5,beta=0.5': '(0.5, 0.5)', 'alpha=0.2,beta=0.8': '(0.2, 0.8)'}
DS_NAME = {'mmlu_pro': 'MMLU-Pro', 'supergpqa': 'SuperGPQA', 'livebench': 'LiveBench'}
fixed = rep['fixed']
S = rep['settings']
esc = html.escape


def money(x, digits=None):
    if digits is None:
        digits = 4 if x < 0.01 else 3
    return f"${x:.{digits}f}"


def f3(x):
    return f"{x:.3f}".replace('0.', '.', 1) if 0 <= x < 1 else f"{x:.3f}"


# ---------------------------------------------------------------- charts
def scatter(points, labels, title):
    """points: list of dicts {x: cost, y: score, cls, tip, r}; labels: list of (x, y, text, anchor, dx, dy, cls)."""
    W, H, L, R, T, B = 560, 360, 52, 18, 16, 44
    x0, x1, y0, y1 = math.log10(0.00025), math.log10(0.06), 0.55, 0.92
    px = lambda c: L + (math.log10(c) - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    py = lambda s: T + (y1 - s) / (y1 - y0) * (H - T - B)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart">']
    for s in (0.6, 0.7, 0.8, 0.9):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(s):.1f}" y2="{py(s):.1f}"/>')
        out.append(f'<text class="tick" x="{L - 8}" y="{py(s) + 4:.1f}" text-anchor="end">{s:.1f}</text>')
    for c, lab in ((0.0003, '$0.0003'), (0.001, '$0.001'), (0.003, '$0.003'), (0.01, '$0.01'), (0.03, '$0.03')):
        out.append(f'<line class="grid" x1="{px(c):.1f}" x2="{px(c):.1f}" y1="{T}" y2="{H - B}"/>')
        out.append(f'<text class="tick" x="{px(c):.1f}" y="{H - B + 18}" text-anchor="middle">{lab}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>')
    out.append(f'<text class="axlab" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">每题花费（美元，对数刻度）</text>')
    out.append(f'<text class="axlab" x="14" y="{(T + H - B) / 2:.0f}" text-anchor="middle" '
               f'transform="rotate(-90 14 {(T + H - B) / 2:.0f})">800 道题平均得分</text>')
    for p in points:
        out.append(f'<circle class="{p["cls"]}" cx="{px(p["x"]):.1f}" cy="{py(p["y"]):.1f}" r="{p.get("r", 5)}" '
                   f'data-tip="{esc(p["tip"])}" tabindex="0"/>')
    for x, y, text, anchor, dx, dy, cls in labels:
        out.append(f'<text class="{cls}" x="{px(x) + dx:.1f}" y="{py(y) + dy:.1f}" text-anchor="{anchor}">{esc(text)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def fixed_points(shade_by_model=True):
    pts = []
    for name in NAMES:
        m = fixed[name]
        model = name.split('/')[0]
        pts.append({'x': m['cost'], 'y': m['all'], 'cls': f"pt fixed m-{SHORT[model].replace('-', '')}",
                    'tip': f"固定用 {LABEL[name]}：得分 {m['all']:.3f}，每题 {money(m['cost'])}", 'r': 4.5})
    return pts


FIXED_LABEL_POS = {  # dx, dy, anchor for each fixed configuration's label
    'gpt-5-nano/low': (9, 4, 'start'), 'gpt-5-nano/medium': (-8, 15, 'end'), 'gpt-5-nano/high': (8, 4, 'start'),
    'gpt-5-mini/low': (-8, 4, 'end'), 'gpt-5-mini/medium': (-8, 4, 'end'), 'gpt-5-mini/high': (0, 17, 'middle'),
    'gpt-5/low': (-8, -8, 'end'), 'gpt-5/medium': (0, -11, 'middle'), 'gpt-5/high': (6, 17, 'end'),
}


def fixed_labels(names):
    out = []
    for name in names:
        dx, dy, anchor = FIXED_LABEL_POS[name]
        out.append((fixed[name]['cost'], fixed[name]['all'], LABEL[name], anchor, dx, dy, 'lab fixedlab'))
    return out


def chart_oracle():
    pts = fixed_points()
    labs = fixed_labels(NAMES)
    pos = {'alpha=1.0,beta=0.0': (9, 4, 'start'), 'alpha=0.5,beta=0.5': (-9, 4, 'end'),
           'alpha=0.2,beta=0.8': (9, 4, 'start')}
    for k, v in S.items():
        m = v['references']['oracle']
        pts.append({'x': m['cost'], 'y': m['all'], 'cls': 'pt oracle',
                    'tip': f"理想路由（{SETTING_NAME[k]}）：得分 {m['all']:.3f}，每题 {money(m['cost'])}"})
        dx, dy, a = pos[k]
        labs.append((m['cost'], m['all'], f"理想路由 · {SETTING_NAME[k]}", a, dx, dy, 'lab oraclelab'))
    return scatter(pts, labs, '固定配置与理想路由的得分和每题花费')


def chart_mlp():
    pts = fixed_points()
    labs = fixed_labels(['gpt-5-nano/low', 'gpt-5-mini/low', 'gpt-5/high'])
    for k, v in S.items():
        m = v['routers']['mlp']['mean']
        sd = v['routers']['mlp']['sd']
        pts.append({'x': m['cost'], 'y': m['all'], 'cls': 'pt mlp',
                    'tip': f"MLP 路由（{SETTING_NAME[k]}）：得分 {m['all']:.3f} ± {sd['all']:.3f}，每题 {money(m['cost'])}"})
        g = v['references']['per-group best']
        pts.append({'x': g['cost'], 'y': g['all'], 'cls': 'pt group',
                    'tip': f"按题组选（{SETTING_NAME[k]}）：得分 {g['all']:.3f}，每题 {money(g['cost'])}"})
    mq, mb, ms = (S[k]['routers']['mlp']['mean'] for k in S)
    gq, gb, gs = (S[k]['references']['per-group best'] for k in S)
    labs += [(mq['cost'], mq['all'], 'MLP · 质量优先 / 均衡', 'start', 9, 4, 'lab mlplab'),
             (ms['cost'], ms['all'], 'MLP · 省钱优先', 'middle', 0, -11, 'lab mlplab'),
             (gq['cost'], gq['all'], '按题组 · 质量优先', 'middle', 0, 18, 'lab grouplab'),
             (gb['cost'], gb['all'], '按题组 · 均衡', 'end', -9, 4, 'lab grouplab'),
             (gs['cost'], gs['all'], '= 按题组 · 省钱优先', 'start', 9, 17, 'lab grouplab')]
    return scatter(pts, labs, 'MLP 路由、按题组选与固定配置的得分和每题花费')


def chart_effort():
    W, H, L, R, T, B = 560, 300, 52, 112, 16, 40
    y0, y1 = 0.55, 0.85
    xs = [L + 30, L + 30 + (W - L - R - 60) / 2, W - R - 30]
    py = lambda s: T + (y1 - s) / (y1 - y0) * (H - T - B)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="三个模型在三个推理档位下的平均得分" class="chart">']
    for s in (0.6, 0.7, 0.8):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R + 10}" y1="{py(s):.1f}" y2="{py(s):.1f}"/>')
        out.append(f'<text class="tick" x="{L - 8}" y="{py(s) + 4:.1f}" text-anchor="end">{s:.1f}</text>')
    for x, e in zip(xs, data.EFFORTS):
        out.append(f'<text class="tick" x="{x:.1f}" y="{H - B + 20}" text-anchor="middle">{e}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R + 10}" y1="{H - B}" y2="{H - B}"/>')
    out.append(f'<text class="axlab" x="14" y="{(T + H - B) / 2:.0f}" text-anchor="middle" '
               f'transform="rotate(-90 14 {(T + H - B) / 2:.0f})">800 道题平均得分</text>')
    for model in data.MODELS:
        cls = SHORT[model].replace('-', '')
        ys = [fixed[f"{model}/{e}"]['all'] for e in data.EFFORTS]
        d = ' '.join(f"{'M' if i == 0 else 'L'}{x:.1f},{py(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
        out.append(f'<path class="line m-{cls}" d="{d}"/>')
        for x, y, e in zip(xs, ys, data.EFFORTS):
            m = fixed[f"{model}/{e}"]
            tip = f"{SHORT[model]} {e}：得分 {m['all']:.3f}，每题 {money(m['cost'])}"
            out.append(f'<circle class="pt fixed m-{cls}" cx="{x:.1f}" cy="{py(y):.1f}" r="4.5" tabindex="0" '
                       f'data-tip="{esc(tip)}"/>')
        out.append(f'<text class="lab endlab" x="{xs[-1] + 12:.1f}" y="{py(ys[-1]) + 4:.1f}">{SHORT[model]} {ys[-1]:.3f}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def label_share(alpha, beta):
    cnt, tot = collections.Counter(), 0
    for f in range(data.FOLDS):
        tr = t.fold != f
        R = data.Normalizer(t.P[tr], t.C[tr]).reward(t.P[tr], t.C[tr], alpha, beta)
        lab = data.best(R, t.C[tr])
        cnt.update(NAMES[k] for k in lab)
        tot += len(lab)
    return {k: v / tot for k, v in cnt.items()}


LABEL_SHARE = {k: label_share(a, b) for k, (a, b) in zip(S, data.SETTINGS)}


def chart_picks():
    key = 'alpha=1.0,beta=0.0'
    lab, mlp = LABEL_SHARE[key], S[key]['routers']['mlp']['pick_share']
    rows = [('gpt-5-nano/low', 'nano low'), ('gpt-5-mini/low', 'mini low'), ('gpt-5-nano/medium', 'nano medium')]
    other = [n for n in NAMES if n not in dict(rows)]
    vals = [(name, lab.get(k, 0), mlp.get(k, 0)) for k, name in rows]
    vals.append(('其余 6 个配置', sum(lab.get(k, 0) for k in other), sum(mlp.get(k, 0) for k in other)))
    W, L, R, T = 560, 112, 56, 8
    bh, gap, row = 13, 3, 46
    H = T + row * len(vals) + 30
    px = lambda v: L + v / 0.8 * (W - L - R)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="训练标签与 MLP 选择在各配置上的占比" class="chart">']
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T}" y2="{H - 26}"/>')
        out.append(f'<text class="tick" x="{px(v):.1f}" y="{H - 8}" text-anchor="middle">{v:.0%}</text>')
    for i, (name, a, b) in enumerate(vals):
        y = T + i * row + 8
        out.append(f'<text class="lab rowlab" x="{L - 10}" y="{y + bh + 2:.1f}" text-anchor="end">{esc(name)}</text>')
        for j, (v, cls, who) in enumerate(((a, 'bar labelbar', '训练标签'), (b, 'bar mlpbar', 'MLP 的选择'))):
            yy = y + j * (bh + gap)
            w = max(px(v) - L, 1)
            out.append(f'<rect class="{cls}" x="{L}" y="{yy:.1f}" width="{w:.1f}" height="{bh}" rx="3" tabindex="0" '
                       f'data-tip="{esc(f"{name}：{who} {v:.1%}")}"/>')
            out.append(f'<text class="val" x="{L + w + 6:.1f}" y="{yy + bh - 2:.1f}">{v:.0%}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{H - 26}"/>')
    out.append('</svg>')
    return '\n'.join(out)


# ---------------------------------------------------------------- tables
def table(head, rows, cls='num'):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


cell = collections.defaultdict(list)
for s in scores:
    cell[(s['task_group'], s['model'], s['effort'])].append(s['score'])
tot_cost = collections.Counter()
for s in scores:
    tot_cost[f"{s['model']}/{s['effort']}"] += s['cost_usd']
total_cost = sum(tot_cost.values())

fixed_rows = [[LABEL[n], *(f"{fixed[n][d]:.3f}" for d in data.DATASETS), f"<b>{fixed[n]['all']:.3f}</b>",
               money(fixed[n]['cost']), f"${tot_cost[n]:.2f}"] for n in NAMES]
fixed_rows.append(['随机（9 个配置的期望）', *(f"{fixed['random'][d]:.3f}" for d in data.DATASETS),
                   f"<b>{fixed['random']['all']:.3f}</b>", money(fixed['random']['cost']), '—'])

groups = list(manifest['groups'])
group_rows = []
for g in groups:
    vals = [np.mean(cell[(g, m, e)]) for m, e in data.CONFIGS]
    top = max(vals)
    ds, name = g.split('/', 1)
    group_rows.append([f"{DS_NAME[ds]} · {name.replace('_', ' ')}"] +
                      [f"<b>{v:.2f}</b>" if v == top else f"{v:.2f}" for v in vals])

TASK_NAME = {'AMPS_Hard': 'AMPS', 'math_comp': '数学竞赛', 'olympiad': '奥赛', 'spatial': '空间推理',
             'web_of_lies_v2': '真假话推理', 'zebra_puzzle': '斑马谜题', 'LCB_generation': '写程序',
             'coding_completion': '补全代码', 'cta': '列类型标注', 'tablejoin': '表连接', 'tablereformat': '表格式转换',
             'paraphrase': '改写', 'simplify': '简化', 'story_generation': '续写故事', 'summarize': '摘要'}
strata_rows = []
for g, v in manifest['groups'].items():
    ds, name = g.split('/', 1)
    if ds == 'supergpqa':
        how = 'easy / middle / hard 各 20'
    elif ds == 'livebench':
        c = collections.Counter()
        for k, n in v['strata_sampled'].items():
            c[k.split('/')[0]] += n
        how = '，'.join(f"{TASK_NAME.get(k, k)} {n}" for k, n in c.items())
    else:
        src = collections.Counter()
        for k, n in v['strata_sampled'].items():
            src[k.split('-')[0]] += n
        names = {'ori_mmlu': '原 MMLU', 'scibench': 'SciBench', 'theoremQA': 'TheoremQA', 'stemez': 'STEM-EZ'}
        how = '，'.join(f"{names.get(k, k)} {n}" for k, n in src.items())
    strata_rows.append([f"{DS_NAME[ds]} · {name.replace('_', ' ')}", f"{v['pool_after_dedup']:,}", str(v['sampled']), how])

route_rows = []
for k, v in S.items():
    m, sd = v['routers']['mlp']['mean'], v['routers']['mlp']['sd']
    route_rows.append([f"<b>{SETTING_NAME[k]}</b> {SETTING_AB[k]}", '', '', '', '', '', ''])
    route_rows.append(['MLP 路由（5 个种子）', *(f"{m[d]:.3f} ± {sd[d]:.3f}" for d in data.DATASETS),
                       f"{m['macro']:.3f} ± {sd['macro']:.3f}", f"<b>{m['all']:.3f}</b> ± {sd['all']:.3f}", money(m['cost'], 5)])
    for ref, nm in (('per-group best', '按题组选'), ('oracle', '理想路由')):
        r = v['references'][ref]
        route_rows.append([nm, *(f"{r[d]:.3f}" for d in data.DATASETS), f"{r['macro']:.3f}", f"<b>{r['all']:.3f}</b>",
                           money(r['cost'], 5)])

diff_rows = []
for k, v in S.items():
    vs = v['routers']['mlp']['vs']
    a, b = vs['per-group best'], vs['gpt-5/high']
    diff_rows.append([SETTING_NAME[k], f"{a[0]:+.3f}（{a[1]:+.3f} 到 {a[2]:+.3f}）", f"{b[0]:+.3f}（{b[1]:+.3f} 到 {b[2]:+.3f}）"])

pick_rows = []
for n in NAMES:
    pick_rows.append([LABEL[n]] + [f"{LABEL_SHARE[k].get(n, 0):.0%} / {S[k]['routers']['mlp']['pick_share'].get(n, 0):.0%}" for k in S])

acc = np.mean([j['train_label_accuracy'] for j in picks_jobs])
tok_in = sum(s['input_tokens'] for s in scores)
tok_out = sum(s['output_tokens'] for s in scores)
tok_reason = sum(s['reasoning_tokens'] for s in scores)
trunc = sum(s['truncated'] for s in scores)
overhead = S['alpha=1.0,beta=0.0']['routers']['mlp']['overhead_ms_per_query']

q, mlpq = S['alpha=1.0,beta=0.0']['references']['oracle'], S['alpha=1.0,beta=0.0']['routers']['mlp']['mean']
g5h, minil, nanol = fixed['gpt-5/high'], fixed['gpt-5-mini/low'], fixed['gpt-5-nano/low']
g5l = fixed['gpt-5/low']
lab_nano = LABEL_SHARE['alpha=1.0,beta=0.0']['gpt-5-nano/low']
mlp_nano = S['alpha=1.0,beta=0.0']['routers']['mlp']['pick_share']['gpt-5-nano/low']
mlp_nano_range = sorted(S[k]['routers']['mlp']['pick_share']['gpt-5-nano/low'] for k in S)

ctx = dict(
    total_cost=f"{total_cost:.2f}", n_calls=f"{len(scores):,}",
    size_gain=f"{g5l['all'] - nanol['all']:+.2f}", effort_gain=f"{g5h['all'] - g5l['all']:+.2f}",
    oracle_score=f"{q['all']:.3f}", oracle_cost=money(q['cost']), g5h_score=f"{g5h['all']:.3f}", g5h_cost=money(g5h['cost'], 3),
    cost_ratio=f"{g5h['cost'] / q['cost']:.0f}",
    mlp_score=f"{mlpq['all']:.3f}", mlp_cost=money(mlpq['cost']), mini_score=f"{minil['all']:.3f}", mini_cost=money(minil['cost']),
    lab_nano=f"{lab_nano:.0%}", mlp_nano=f"{mlp_nano:.0%}", mlp_nano_lo=f"{mlp_nano_range[0]:.0%}", mlp_nano_hi=f"{mlp_nano_range[-1]:.0%}",
    acc=f"{acc:.1%}",
    chart_effort=chart_effort(), chart_oracle=chart_oracle(), chart_mlp=chart_mlp(), chart_picks=chart_picks(),
    fixed_table=table(['配置', 'MMLU-Pro<br>250 题', 'SuperGPQA<br>300 题', 'LiveBench<br>250 题', '全部<br>800 题', '每题花费', '800 题总花费'], fixed_rows),
    group_table=table(['题组', *(LABEL[n] for n in NAMES)], group_rows, 'num tight'),
    strata_table=table(['题组', '去重后可选', '抽取', '组内分层（抽到的题数）'], strata_rows, 'textcol'),
    route_table=table(['方法', 'MMLU-Pro', 'SuperGPQA', 'LiveBench', '三个数据集平均', '全部 800 题', '每题花费'], route_rows, 'num route'),
    diff_table=table(['设置', 'MLP − 按题组选', 'MLP − 固定用 gpt-5 high'], diff_rows),
    pick_table=table(['配置', '质量优先', '均衡', '省钱优先'], pick_rows),
    tok_in=f"{tok_in / 1e4:,.0f}", tok_out=f"{tok_out / 1e4:,.0f}", tok_reason=f"{tok_reason / 1e4:,.0f}", trunc=trunc,
    emb_ms=f"{overhead['embedding']:.0f}", router_ms=f"{overhead['router']:.3f}",
)
page = Path(__file__).with_name('report_template.html').read_text()
for k, v in ctx.items():
    page = page.replace('{{' + k + '}}', str(v))
assert '{{' not in page, page[page.index('{{'):page.index('{{') + 40]
OUT.write_text(page)
print('wrote', OUT, len(page), 'bytes')
