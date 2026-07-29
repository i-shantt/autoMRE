"""
AutoRepro-Min: Learned Removability Oracle
Training and Inference

Trains a gradient-boosted classifier on the Phase 4b attempt log and
wraps it for use as a prior inside MultiFileDebugger.

Model choice
------------
HistGradientBoostingClassifier, for three reasons that all matter more
than raw accuracy on a dataset this small: it handles unscaled,
mixed-type columns without preprocessing; it predicts in well under a
millisecond, which it has to, since the whole point is to spend less
time than the validation query it replaces; and it stays interpretable
enough that permutation importances are worth reporting.

Evaluation
----------
Leave-one-repo-out. Only two repos are available, so this gives two
folds — few, but honest. The rejected alternative was leave-one-task-out,
which looks better because it yields more folds while quietly leaking:
four of the six benchmark tasks reduce the same requests checkout, so
per-task folds would train and test on the same files.

The headline numbers are deliberately the two operating points the
integration actually uses, not a generic score:

    precision at p>0.9   — of units the oracle calls safe at the Phase 4a
                           threshold, how many really were. A false
                           positive here means a bulk prune that rolls
                           back, costing a wasted query.
    precision at p<0.1   — of units the oracle calls hopeless at the
                           Phase 4b threshold, how many really were. A
                           false negative here means a removable unit is
                           never attempted, so the output gets worse.

A model can post a respectable AUC and still be useless at both ends,
which is why AUC is reported but not led with.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .features import FEATURE_NAMES, UnitFeatures, extract_features

MODEL_FILENAME = "removability_model.pkl"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME

# Operating points used by the MultiFileDebugger integration.
PHASE_4A_SAFE_THRESHOLD = 0.9
PHASE_4B_SKIP_THRESHOLD = 0.1


# ------------------------------------------------------------- dataset

@dataclass
class Dataset:
    X: List[List[float]]
    y: List[int]
    groups: List[str]
    task_ids: List[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_safe(self) -> int:
        return sum(self.y)

    @property
    def n_unsafe(self) -> int:
        return len(self.y) - self.n_safe


def load_dataset(path: Path | str,
                 task_prefix: Optional[str] = None,
                 exclude_task_prefix: Optional[str] = None) -> Dataset:
    """Read the JSONL attempt log into aligned feature/label lists.

    Column order comes from FEATURE_NAMES rather than from the row's own
    key order, so a row written by an older feature set fails loudly
    instead of silently shifting columns.
    """
    path = Path(path)
    X: List[List[float]] = []
    y: List[int] = []
    groups: List[str] = []
    task_ids: List[str] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = row.get("task_id") or ""
            if task_prefix and not task.startswith(task_prefix):
                continue
            if exclude_task_prefix and task.startswith(exclude_task_prefix):
                continue
            missing = [c for c in FEATURE_NAMES if c not in row]
            if missing:
                raise ValueError(
                    f"{path}:{lineno} is missing {len(missing)} feature "
                    f"column(s), first few: {missing[:5]}. The log was "
                    f"probably written by a different feature version — "
                    f"regenerate it.")
            X.append([float(row[c]) for c in FEATURE_NAMES])
            y.append(int(row["was_safely_removable"]))
            groups.append(row.get("repo_slug") or "<unknown>")
            task_ids.append(row.get("task_id") or "<unknown>")

    return Dataset(X=X, y=y, groups=groups, task_ids=task_ids)


# ------------------------------------------------------------ training

def _precision_at(y_true: Sequence[int], probs: Sequence[float],
                  threshold: float, above: bool) -> Tuple[float, int]:
    """Precision of the decision this threshold actually drives.

    above=True  -> "predicted safe": fraction of p>=threshold that were safe.
    above=False -> "predicted hopeless": fraction of p<threshold that were
                   in fact unsafe.
    Returns (precision, n_selected); precision is nan when nothing is
    selected, which is itself a useful signal.
    """
    if above:
        picked = [t for t, p in zip(y_true, probs) if p >= threshold]
        hits = sum(picked)
    else:
        picked = [t for t, p in zip(y_true, probs) if p < threshold]
        hits = sum(1 for t in picked if t == 0)

    if not picked:
        return float("nan"), 0
    return hits / len(picked), len(picked)


def _precision_at_recall(y_true: Sequence[int], probs: Sequence[float],
                         target_recall: float = 0.5) -> float:
    """Best precision achievable at or above `target_recall` for class 1."""
    from sklearn.metrics import precision_recall_curve

    if len(set(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(list(y_true), list(probs))
    ok = [p for p, r in zip(precision, recall) if r >= target_recall]
    return max(ok) if ok else float("nan")


def _build_model(class_weight: Optional[str] = "balanced"):
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Small dataset: shallow trees and a low iteration cap do more good
    # than tuning. early_stopping off because the validation split it
    # would carve out is a meaningful slice of the data here.
    kwargs: Dict[str, Any] = dict(
        max_iter=200,
        max_depth=4,
        learning_rate=0.06,
        min_samples_leaf=8,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    try:
        return HistGradientBoostingClassifier(class_weight=class_weight,
                                              **kwargs)
    except TypeError:
        # class_weight predates some sklearn versions; imbalance handling
        # is a nice-to-have, not a requirement.
        return HistGradientBoostingClassifier(**kwargs)


def cross_validate(data: Dataset) -> Dict[str, Any]:
    """Leave-one-repo-out evaluation."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    X = np.asarray(data.X, dtype=float)
    y = np.asarray(data.y, dtype=int)
    groups = np.asarray(data.groups)

    unique_groups = sorted(set(data.groups))
    folds: List[Dict[str, Any]] = []
    pooled_true: List[int] = []
    pooled_prob: List[float] = []

    for held_out in unique_groups:
        test_mask = groups == held_out
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(set(y[train_mask].tolist())) < 2:
            folds.append({"held_out_repo": held_out,
                          "skipped": "training fold has a single class"})
            continue

        model = _build_model()
        model.fit(X[train_mask], y[train_mask])
        probs = model.predict_proba(X[test_mask])[:, 1]

        y_test = y[test_mask].tolist()
        pooled_true.extend(y_test)
        pooled_prob.extend(probs.tolist())

        p4a, n4a = _precision_at(y_test, probs, PHASE_4A_SAFE_THRESHOLD,
                                 above=True)
        p4b, n4b = _precision_at(y_test, probs, PHASE_4B_SKIP_THRESHOLD,
                                 above=False)
        folds.append({
            "held_out_repo": held_out,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "test_safe_rate": float(np.mean(y_test)) if y_test else None,
            "auc": (float(roc_auc_score(y_test, probs))
                    if len(set(y_test)) > 1 else None),
            "precision_at_recall_0.5": _precision_at_recall(y_test, probs),
            "precision_above_0.9": None if n4a == 0 else float(p4a),
            "n_above_0.9": n4a,
            "precision_below_0.1": None if n4b == 0 else float(p4b),
            "n_below_0.1": n4b,
        })

    pooled: Dict[str, Any] = {}
    if pooled_true and len(set(pooled_true)) > 1:
        p4a, n4a = _precision_at(pooled_true, pooled_prob,
                                 PHASE_4A_SAFE_THRESHOLD, above=True)
        p4b, n4b = _precision_at(pooled_true, pooled_prob,
                                 PHASE_4B_SKIP_THRESHOLD, above=False)
        pooled = {
            "auc": float(roc_auc_score(pooled_true, pooled_prob)),
            "precision_at_recall_0.5": _precision_at_recall(pooled_true,
                                                            pooled_prob),
            "precision_above_0.9": None if n4a == 0 else float(p4a),
            "n_above_0.9": n4a,
            "precision_below_0.1": None if n4b == 0 else float(p4b),
            "n_below_0.1": n4b,
            "base_rate_safe": float(np.mean(pooled_true)),
        }

    return {"folds": folds, "pooled_out_of_fold": pooled}


def permutation_importances(data: Dataset, top_n: int = 15) -> List[Dict]:
    """Which columns the final model actually leans on."""
    import numpy as np
    from sklearn.inspection import permutation_importance

    X = np.asarray(data.X, dtype=float)
    y = np.asarray(data.y, dtype=int)
    if len(set(data.y)) < 2:
        return []

    model = _build_model()
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=10,
                                    random_state=0, scoring="roc_auc")
    ranked = sorted(
        ({"feature": name,
          "importance": float(result.importances_mean[i]),
          "std": float(result.importances_std[i])}
         for i, name in enumerate(FEATURE_NAMES)),
        key=lambda d: -d["importance"])
    return ranked[:top_n]


def train(data_path: Path | str,
          model_out: Path | str = DEFAULT_MODEL_PATH,
          report_out: Optional[Path | str] = None,
          task_prefix: Optional[str] = None,
          exclude_task_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Cross-validate, fit on everything, and write the model."""
    import numpy as np

    data = load_dataset(data_path, task_prefix, exclude_task_prefix)
    if len(data) == 0:
        raise ValueError(f"{data_path} has no rows")

    report: Dict[str, Any] = {
        "n_rows": len(data),
        "n_safe": data.n_safe,
        "n_unsafe": data.n_unsafe,
        "safe_rate": data.n_safe / len(data),
        "repos": sorted(set(data.groups)),
        "n_tasks": len(set(data.task_ids)),
        "feature_count": len(FEATURE_NAMES),
    }

    if len(set(data.y)) < 2:
        report["error"] = (
            "only one class present — every Phase 4b attempt had the same "
            "outcome, so there is nothing to learn")
        if report_out:
            Path(report_out).write_text(json.dumps(report, indent=2))
        return report

    report["cv"] = cross_validate(data)
    report["importances"] = permutation_importances(data)

    final = _build_model()
    final.fit(np.asarray(data.X, dtype=float),
              np.asarray(data.y, dtype=int))

    payload = {
        "model": final,
        "feature_names": list(FEATURE_NAMES),
        "trained_on_rows": len(data),
        "repos": sorted(set(data.groups)),
    }
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as fh:
        pickle.dump(payload, fh)
    report["model_path"] = str(model_out)

    if report_out:
        Path(report_out).write_text(json.dumps(report, indent=2))
    return report


# ----------------------------------------------------------- inference

class LearnedRemovabilityOracle:
    """Predicts p(removing this unit keeps the bug reproducing)."""

    def __init__(self, model: Any, feature_names: Sequence[str]):
        if list(feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                "model was trained on a different feature layout "
                f"({len(feature_names)} columns vs {len(FEATURE_NAMES)}); "
                "retrain it against the current features.py")
        self.model = model
        self.feature_names = list(feature_names)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH
             ) -> "LearnedRemovabilityOracle":
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["model"], payload["feature_names"])

    @classmethod
    def load_if_available(cls, path: Path | str = DEFAULT_MODEL_PATH
                          ) -> Optional["LearnedRemovabilityOracle"]:
        """Load, or return None if unavailable — callers degrade to the
        plain heuristic rather than failing."""
        try:
            return cls.load(path)
        except Exception:
            return None

    def predict(self, features: UnitFeatures) -> float:
        return self.predict_batch([features])[0]

    def predict_batch(self, features: Sequence[UnitFeatures]) -> List[float]:
        if not features:
            return []
        import numpy as np
        X = np.asarray([f.to_vector() for f in features], dtype=float)
        return self.model.predict_proba(X)[:, 1].tolist()

    def predict_node(self, node: Any, source: str, executed_lines,
                     file_path=None) -> float:
        """Convenience path straight from a tree-sitter node."""
        return self.predict(
            extract_features(node, source, executed_lines, file_path))


def _format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"rows          : {report['n_rows']}")
    lines.append(f"safe / unsafe : {report['n_safe']} / {report['n_unsafe']}"
                 f"  ({report['safe_rate']*100:.1f}% safe)")
    lines.append(f"repos         : {', '.join(report['repos'])}")
    lines.append(f"tasks         : {report['n_tasks']}")
    lines.append(f"features      : {report['feature_count']}")

    if "error" in report:
        lines.append("")
        lines.append(f"NOT TRAINED: {report['error']}")
        return "\n".join(lines)

    lines.append("")
    lines.append("leave-one-repo-out:")
    for fold in report["cv"]["folds"]:
        if "skipped" in fold:
            lines.append(f"  hold out {fold['held_out_repo']}: "
                         f"skipped ({fold['skipped']})")
            continue
        auc = fold["auc"]
        lines.append(
            f"  hold out {fold['held_out_repo']:<18} "
            f"n={fold['n_test']:<5} "
            f"safe={fold['test_safe_rate']*100:5.1f}%  "
            f"auc={'n/a' if auc is None else f'{auc:.3f}'}")
        lines.append(
            f"      p>0.9 precision: "
            f"{_fmt_pct(fold['precision_above_0.9'])} "
            f"(n={fold['n_above_0.9']})   "
            f"p<0.1 precision: "
            f"{_fmt_pct(fold['precision_below_0.1'])} "
            f"(n={fold['n_below_0.1']})")

    pooled = report["cv"].get("pooled_out_of_fold") or {}
    if pooled:
        lines.append("")
        lines.append("pooled out-of-fold:")
        lines.append(f"  auc                 : {pooled['auc']:.3f}")
        lines.append(f"  base rate (safe)    : "
                     f"{pooled['base_rate_safe']*100:.1f}%")
        lines.append(f"  precision @ p>0.9   : "
                     f"{_fmt_pct(pooled['precision_above_0.9'])} "
                     f"(n={pooled['n_above_0.9']})")
        lines.append(f"  precision @ p<0.1   : "
                     f"{_fmt_pct(pooled['precision_below_0.1'])} "
                     f"(n={pooled['n_below_0.1']})")

    if report.get("importances"):
        lines.append("")
        lines.append("top permutation importances:")
        for imp in report["importances"][:10]:
            lines.append(f"  {imp['feature']:<32} "
                         f"{imp['importance']:+.4f}")
    return "\n".join(lines)


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value != value:  # nan
        return "n/a"
    return f"{value*100:.1f}%"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the learned removability oracle.")
    parser.add_argument("--data", required=True,
                        help="Path to oracle_training_data.jsonl")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--task-prefix", default=None,
        help="Train only on rows whose task_id starts with this. Use to "
             "hold the scored tasks out entirely.")
    parser.add_argument("--exclude-task-prefix", default=None,
        help="Drop rows whose task_id starts with this.")
    args = parser.parse_args()

    report = train(args.data, args.model_out, args.report_out,
                   args.task_prefix, args.exclude_task_prefix)
    print(_format_report(report))
    if "error" not in report:
        print()
        print(f"model -> {report['model_path']}")
    if args.report_out:
        print(f"report -> {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
