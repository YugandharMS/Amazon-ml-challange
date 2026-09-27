"""
Model Training, Threshold Calibration, and Competitive Assignment.
Uses HistGradientBoostingClassifier to optimize macro F0.5.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from features import FEATURE_COLUMNS


def train_classifier(
    train_df: pd.DataFrame,
) -> HistGradientBoostingClassifier:
    """Train HistGradientBoostingClassifier on features."""
    X = train_df[FEATURE_COLUMNS].fillna(0.0)
    y = train_df["label"].values

    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def compute_macro_f05(
    gt_dict: dict[str, set[str]],
    pred_dict: dict[str, set[str]],
    all_s1_ids: list[str],
) -> float:
    """Calculate macro-averaged F0.5 score across all Source 1 entities."""
    scores = []
    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        n_true = len(true_set)
        n_pred = len(pred_set)

        if n_true == 0 and n_pred == 0:
            scores.append(1.0)
        elif n_true == 0 or n_pred == 0:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            if tp == 0:
                scores.append(0.0)
            else:
                p = tp / n_pred
                r = tp / n_true
                f05 = (1.25 * p * r) / (0.25 * p + r)
                scores.append(f05)
    return float(np.mean(scores))


def calibrate_threshold(
    val_df_s2: pd.DataFrame,
    val_df_s3: pd.DataFrame,
    val_gt_dict: dict[str, set[str]],
    val_s1_ids: list[str],
    clf: HistGradientBoostingClassifier,
) -> tuple[float, float]:
    """Search for the threshold that maximizes macro F0.5 with 1-to-1 competitive assignment."""
    val_df_s2 = val_df_s2.copy()
    val_df_s3 = val_df_s3.copy()

    val_df_s2["prob"] = clf.predict_proba(val_df_s2[FEATURE_COLUMNS].fillna(0.0))[:, 1]
    val_df_s3["prob"] = clf.predict_proba(val_df_s3[FEATURE_COLUMNS].fillna(0.0))[:, 1]

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_th = 0.50
    best_score = -1.0

    for th in thresholds:
        # Filter and apply competitive 1-to-1 matching per target
        p2 = val_df_s2[val_df_s2["prob"] >= th].sort_values("prob", ascending=False).drop_duplicates(subset=["matched_entity_id"])
        p3 = val_df_s3[val_df_s3["prob"] >= th].sort_values("prob", ascending=False).drop_duplicates(subset=["matched_entity_id"])

        preds = pd.concat([p2[["source1_entity_id", "matched_entity_id"]], p3[["source1_entity_id", "matched_entity_id"]]])
        pred_dict = {
            s1_id: set(grp["matched_entity_id"].values)
            for s1_id, grp in preds.groupby("source1_entity_id")
        }
        score = compute_macro_f05(val_gt_dict, pred_dict, val_s1_ids)
        if score > best_score:
            best_score = score
            best_th = th

    return best_th, best_score
