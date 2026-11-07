"""Routing table from a scored baseline run, the proposal's quality-cost reward, and the non-learned references.

Each question has a score p and a cost c (USD at list prices) under each of the 9 configurations. The reward is
r = alpha * p~ - beta * c~ (proposal eq. 3), with p~ and c~ min-max normalized on the training questions. Cost is
normalized on a log scale: raw costs span four orders of magnitude, so a linear min-max leaves every response but
the few most expensive ones near 0 and the three (alpha, beta) settings would pick alike. A question's label is its
highest-reward configuration; ties (only possible at beta = 0) go to the cheaper response, then the earlier
configuration. Folds are the data set's own (rank % 5): each question is tested once over the 5 folds.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline'))
from run import DEFAULT_SET, EFFORTS, MODELS  # noqa: E402
from trapi import RUNS  # noqa: E402

CONFIGS = [(m, e) for m in MODELS for e in EFFORTS]
SETTINGS = [(1.0, 0.0), (0.5, 0.5), (0.2, 0.8)]
DATASETS = ['mmlu_pro', 'supergpqa', 'livebench']
FOLDS = 5


@dataclass
class Table:
    ids: list          # sample_id per question, in data-set order
    dataset: np.ndarray
    group: np.ndarray  # task group, e.g. 'mmlu_pro/law'
    fold: np.ndarray
    P: np.ndarray      # (questions, configs) benchmark score
    C: np.ndarray      # (questions, configs) cost in USD


def load(run='full', set_path=DEFAULT_SET):
    rows = [json.loads(line) for line in open(set_path)]
    index = {r['sample_id']: i for i, r in enumerate(rows)}
    P = np.full((len(rows), len(CONFIGS)), np.nan)
    C = np.full_like(P, np.nan)
    for line in open(RUNS / run / 'scores.jsonl'):
        s = json.loads(line)
        k = CONFIGS.index((s['model'], s['effort']))
        P[index[s['sample_id']], k], C[index[s['sample_id']], k] = s['score'], s['cost_usd']
    assert not np.isnan(P).any() and not np.isnan(C).any(), 'every question needs all 9 configurations scored'
    return Table([r['sample_id'] for r in rows], np.array([r['dataset'] for r in rows]),
                 np.array([r['task_group'] for r in rows]), np.array([r['fold'] for r in rows]), P, C)


class Normalizer:
    """Min-max of p and of log c over the training questions' responses."""

    def __init__(self, P, C):
        self.p = P.min(), P.max()
        self.c = np.log(C).min(), np.log(C).max()

    def reward(self, P, C, alpha, beta):
        p = (P - self.p[0]) / (self.p[1] - self.p[0])
        c = (np.log(C) - self.c[0]) / (self.c[1] - self.c[0])
        return alpha * p - beta * c


def best(R, C):
    """Per row, the highest-reward configuration; ties go to the cheaper response, then the earlier configuration."""
    k = np.arange(R.shape[1])
    return np.array([np.lexsort((k, c, -r))[0] for r, c in zip(R, C)])


def per_group_best(table, train, R_train):
    """Per task group, the configuration with the highest mean training reward (ties: lower mean cost, then order)."""
    picks = {}
    for g in np.unique(table.group[train]):
        rows = table.group[train] == g
        picks[g] = best(R_train[rows].mean(0, keepdims=True), table.C[train][rows].mean(0, keepdims=True))[0]
    return picks
