"""
Unit tests for Anveshak metrics, IoU calculations, and the conflict resolution rule.
"""

import numpy as np
import pandas as pd
from src.utils.metrics import calculate_iou, resolve_conflicts, evaluate_oe_score


def test_conflict_rule():
    # Test case 1: Known box vs Unknown box overlapping with IoU >= 0.5
    # Known box has higher score -> Unknown box should be suppressed
    df = pd.DataFrame([
        {
            "image_id": "img1",
            "class_name": "class_a",
            "score": 0.85,
            "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200
        },
        {
            "image_id": "img1",
            "class_name": "unknown",
            "score": 0.60,
            "xmin": 105, "ymin": 105, "xmax": 200, "ymax": 200  # High IoU with class_a
        },
        {
            "image_id": "img1",
            "class_name": "unknown",
            "score": 0.90,
            "xmin": 300, "ymin": 300, "xmax": 400, "ymax": 400  # Independent unknown
        }
    ])

    resolved = resolve_conflicts(df, iou_thresh=0.5)
    assert len(resolved) == 2, f"Expected 2 rows after conflict resolution, got {len(resolved)}"
    assert "class_a" in resolved['class_name'].values
    assert 0.90 in resolved['score'].values
    assert 0.60 not in resolved['score'].values
    print("[✓] Conflict rule (known > unknown) passed.")

    # Test case 2: Unknown box has higher score -> Known box should be suppressed
    df2 = pd.DataFrame([
        {
            "image_id": "img2",
            "class_name": "class_b",
            "score": 0.40,
            "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150
        },
        {
            "image_id": "img2",
            "class_name": "unknown",
            "score": 0.75,
            "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150
        }
    ])
    resolved2 = resolve_conflicts(df2, iou_thresh=0.5)
    assert len(resolved2) == 1
    assert resolved2.iloc[0]['class_name'] == "unknown"
    assert resolved2.iloc[0]['score'] == 0.75
    print("[✓] Conflict rule (unknown > known) passed.")


def test_oe_score_calculation():
    # Construct synthetic ground truth
    gt_df = pd.DataFrame([
        {"image_id": "img1", "class_name": "ship", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200},
        {"image_id": "img1", "class_name": "unknown", "xmin": 300, "ymin": 300, "xmax": 400, "ymax": 400}
    ])

    # Perfect predictions
    preds_df = pd.DataFrame([
        {"image_id": "img1", "class_name": "ship", "score": 0.95, "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200},
        {"image_id": "img1", "class_name": "unknown", "score": 0.88, "xmin": 300, "ymin": 300, "xmax": 400, "ymax": 400}
    ])

    scores = evaluate_oe_score(preds_df, gt_df, iou_thresh=0.5)
    print(f"[✓] Perfect prediction score result: {scores}")
    assert scores["mAP_K"] == 1.0
    assert scores["U-F1"] == 1.0
    assert scores["OE-Score"] == 100.0


if __name__ == "__main__":
    test_conflict_rule()
    test_oe_score_calculation()
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
