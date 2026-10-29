"""
scripts/train_msm.py
====================
Matcher Selection Model (MSM) Training Pipeline (L1.5 / S4.5).

Trains a LightGBM multi-class meta-model on 13-dimensional feature vectors extracted
from train-split image pairs with oracle best matcher ground truth labels.

Key Requirements:
  - F15 / F27: Strict Geo-Cell Disjoint GroupKFold CV (zero geographic data leakage)
  - Features: 13-D MSMFeatureVector from src.selector.features
  - Target labels: 0=sift (M0), 1=rift2 (M1), 2=lightglue (M2), 3=crater (M3)
  - Outputs: models/msm_v1.pkl and models/msm_v1_stats.json

Usage:
  python scripts/train_msm.py \\
      --manifest data/pairs/manifest.jsonl \\
      --out-model models/msm_v1.pkl \\
      --out-stats models/msm_v1_stats.json \\
      --cv-splits 5 --seed 42

References:
  - FEATURES.md F26, F27
  - ARCHITECTURE.md §3 (L1.5)
  - DECISIONS.md D16, D18
  - PROGRESS.md §5.5.6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.provenance import build_provenance, set_global_seed
from src.selector import (
    FEATURE_NAMES,
    MATCHER_NAMES,
    MATCHER_TO_INDEX,
    MSMFeatureVector,
    extract_features,
    vectorize_features,
)


logger = logging.getLogger("train_msm")


def _get_oracle_label(
    pair_record: Dict[str, Any],
    results_dir: Path,
) -> int:
    """
    Determine the oracle best matcher index (0=sift, 1=rift2, 2=lightglue, 3=crater).

    Looks up evaluation results in results/<pair_id>/ or falls back to domain heuristic.
    """
    pair_id = pair_record.get("pair_id", "")
    pair_dir = results_dir / pair_id

    # 1. Check leaderboard or pair_results if available
    best_matcher: Optional[str] = None
    best_score = float("inf")

    if pair_dir.exists():
        for mid in ["lightglue", "rift2", "sift", "crater"]:
            stats_path = pair_dir / mid / "selection_stats.json"
            if stats_path.exists():
                try:
                    with open(stats_path, "r", encoding="utf-8") as fh:
                        st = json.load(fh)
                    # Higher inlier count / coverage gives proxy if eval not run
                    cov = float(st.get("coverage_after", 0.0))
                    n = int(st.get("n_after", 0))
                    if n >= 25 and cov >= 0.60:
                        # Proxy score: negative inlier count
                        score = -float(n)
                        if score < best_score:
                            best_score = score
                            best_matcher = mid
                except Exception:
                    pass

    if best_matcher and best_matcher in MATCHER_TO_INDEX:
        return MATCHER_TO_INDEX[best_matcher]

    # 2. Heuristic domain rule labeling for bootstrapping/synthetic training
    # - Polar highland with extreme shadow / large delta az -> RIFT2 (M1)
    # - High crater density with valid terrain -> Crater (M3)
    # - General cross-sensor high texture -> LightGlue (M2)
    # - Default baseline -> SIFT (M0)
    c_density = float(pair_record.get("crater_density_per_km2") or 0.0)
    lat = abs(float(pair_record.get("latitude_center_deg") or 0.0))
    delta_az = float(pair_record.get("delta_azimuth_deg") or 0.0)
    terrain = str(pair_record.get("terrain_class") or "").lower()

    if c_density >= 5.0 and terrain in ("polar_highland", "crater_floor", "highland"):
        return 3  # crater
    elif lat >= 50.0 or delta_az >= 40.0:
        return 1  # rift2
    elif terrain in ("equatorial_highland", "equatorial_mare", "highland"):
        return 2  # lightglue
    else:
        return 0  # sift


def load_dataset(
    manifest_path: Path,
    results_dir: Path,
    processed_dir: Path,
    splits: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract (X, y, groups, pair_ids) from manifest and processed metadata.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    pairs: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if not splits or rec.get("split") in splits:
                        pairs.append(rec)
                except json.JSONDecodeError:
                    continue

    if not pairs:
        raise ValueError(f"No valid pairs found in {manifest_path} for splits {splits}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups: List[str] = []
    pair_ids: List[str] = []

    for pair in pairs:
        pair_id = pair.get("pair_id", "unknown")
        meta_path = processed_dir / pair_id / "meta.json"
        meta_json = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta_json = json.load(fh)
            except Exception:
                pass

        feat = extract_features(pair, meta_json)
        x_vec = vectorize_features(feat)
        y_val = _get_oracle_label(pair, results_dir)
        geo_cell = str(pair.get("geo_cell", f"cell_{pair_id}"))

        X_list.append(x_vec)
        y_list.append(y_val)
        groups.append(geo_cell)
        pair_ids.append(pair_id)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y, groups, pair_ids


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train model using geo-cell grouped cross-validation and compute evaluation stats.
    """
    set_global_seed(seed)
    unique_groups = list(set(groups))
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        actual_splits = 2

    # GroupKFold to ensure zero geo-cell leakage (F15)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=actual_splits)

    fold_accuracies: List[float] = []
    fold_top2_accuracies: List[float] = []

    # Verify zero leakage across folds (F15)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_groups = set(groups[i] for i in train_idx)
        val_groups = set(groups[i] for i in val_idx)
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Data leakage detected in fold {fold}: overlapping groups {overlap}")

    # Model training with LightGBM (or scikit-learn fallback)
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=1,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(X, y)
        importances_split = [float(x) for x in model.feature_importances_]
        # Gain importance
        booster = model.booster_
        importances_gain = [float(x) for x in booster.feature_importance(importance_type="gain")]
    except Exception as exc:
        logger.warning("LightGBM fit failed or not available: %s. Using RandomForestClassifier.", exc)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=seed,
        )
        model.fit(X, y)
        importances_split = [float(x) for x in model.feature_importances_]
        importances_gain = importances_split

    # Cross-validation performance
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # Fit fold model
        try:
            import lightgbm as lgb
            fold_model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=4,
                n_estimators=30,
                learning_rate=0.05,
                num_leaves=10,
                min_child_samples=1,
                random_state=seed + fold,
                verbosity=-1,
            )
            fold_model.fit(X_tr, y_tr)
            probs = fold_model.predict_proba(X_va)
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            fold_model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=seed + fold)
            fold_model.fit(X_tr, y_tr)
            probs = fold_model.predict_proba(X_va)
            # Handle missing classes in fold
            if probs.shape[1] < 4:
                full_probs = np.zeros((len(X_va), 4), dtype=np.float32)
                for idx_c, c in enumerate(fold_model.classes_):
                    full_probs[:, c] = probs[:, idx_c]
                probs = full_probs

        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_va))
        fold_accuracies.append(acc)

        top2 = np.argsort(probs, axis=1)[:, -2:]
        top2_acc = float(np.mean([y_va[i] in top2[i] for i in range(len(y_va))]))
        fold_top2_accuracies.append(top2_acc)

    stats: Dict[str, Any] = {
        "model_version": "msm_v1",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "n_classes": len(MATCHER_NAMES),
        "class_names": MATCHER_NAMES,
        "cv_splits": actual_splits,
        "cv_mean_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "cv_mean_top2_accuracy": round(float(np.mean(fold_top2_accuracies)), 4),
        "fold_accuracies": [round(x, 4) for x in fold_accuracies],
        "fold_top2_accuracies": [round(x, 4) for x in fold_top2_accuracies],
        "feature_importance_split": {
            name: round(imp, 4) for name, imp in zip(FEATURE_NAMES, importances_split)
        },
        "feature_importance_gain": {
            name: round(imp, 4) for name, imp in zip(FEATURE_NAMES, importances_gain)
        },
        "geo_cell_leakage_audit": "PASSED (0 overlap between CV folds)",
    }
    return model, stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MSM Training Pipeline (L1.5 / S4.5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default="data/pairs/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--results-dir", default="results", help="Path to benchmark results directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--out-model", default="models/msm_v1.pkl", help="Output path for model pickle")
    parser.add_argument("--out-stats", default="models/msm_v1_stats.json", help="Output path for stats JSON")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of GroupKFold CV splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", nargs="*", default=["train"], help="Splits to train on (e.g. train)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    manifest_path = Path(args.manifest)
    results_dir = Path(args.results_dir)
    processed_dir = Path(args.processed_dir)
    out_model_path = Path(args.out_model)
    out_stats_path = Path(args.out_stats)

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_stats_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training dataset from %s (splits=%s)...", manifest_path, args.splits)
    try:
        X, y, groups, pair_ids = load_dataset(
            manifest_path, results_dir, processed_dir, splits=args.splits,
        )
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return 1

    logger.info(
        "Loaded %d samples across %d distinct geo-cells. Feature matrix shape: %s",
        len(X), len(set(groups)), X.shape,
    )

    logger.info("Training MSM model with %d-fold Geo-Cell Disjoint CV...", args.cv_splits)
    model, stats = train_model(X, y, groups, n_splits=args.cv_splits, seed=args.seed)

    # Embed provenance metadata
    prov = build_provenance()
    stats["provenance"] = prov

    # Save model pickle
    with open(out_model_path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Saved trained MSM model to %s", out_model_path)

    # Save stats JSON
    with open(out_stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved MSM training stats to %s", out_stats_path)

    logger.info(
        "MSM Training Complete: CV Mean Accuracy=%.2f%%, CV Top-2 Accuracy=%.2f%%",
        stats["cv_mean_accuracy"] * 100, stats["cv_mean_top2_accuracy"] * 100,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
