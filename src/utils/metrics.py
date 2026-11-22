"""
Competition Evaluation Metrics & Conflict Resolver for Anveshak OWOD.
Implements:
  - IoU computation & matching (IoU 0.5)
  - Conflict Resolution Rule (IoU >= 0.5 between known and unknown: highest score wins)
  - mAP_K (VOC all-point AP over known classes)
  - Unknown Precision (U-Prec), Recall (U-Rec), F1 (U-F1)
  - OE-Score = 100 * 2 * (mAP_K * U-F1) / (mAP_K + U-F1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) of two boxes [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def resolve_conflicts(predictions_df: pd.DataFrame, iou_thresh: float = 0.5) -> pd.DataFrame:
    """
    Enforces the official competition Conflict Rule:
    'Before scoring, if a known-class prediction and an unknown prediction overlap
    at IoU 0.5 or more, only the one with the higher score counts. Ties go to the earlier row.'
    """
    resolved_rows = []

    for image_id, group in predictions_df.groupby('image_id', sort=False):
        group = group.reset_index(drop=True)
        known_mask = group['class_name'] != 'unknown'
        unknown_mask = group['class_name'] == 'unknown'

        known_indices = group[known_mask].index.tolist()
        unknown_indices = group[unknown_mask].index.tolist()

        suppressed = set()

        for u_idx in unknown_indices:
            u_box = group.loc[u_idx, ['xmin', 'ymin', 'xmax', 'ymax']].to_numpy(dtype=float)
            u_score = group.loc[u_idx, 'score']

            for k_idx in known_indices:
                if k_idx in suppressed:
                    continue
                k_box = group.loc[k_idx, ['xmin', 'ymin', 'xmax', 'ymax']].to_numpy(dtype=float)
                k_score = group.loc[k_idx, 'score']

                if calculate_iou(u_box, k_box) >= iou_thresh:
                    # Conflict found!
                    if u_score > k_score:
                        suppressed.add(k_idx)
                    elif k_score > u_score:
                        suppressed.add(u_idx)
                        break
                    else:
                        # Tie goes to earlier row
                        if u_idx < k_idx:
                            suppressed.add(k_idx)
                        else:
                            suppressed.add(u_idx)
                            break

        valid_group = group.drop(index=list(suppressed))
        resolved_rows.append(valid_group)

    return pd.concat(resolved_rows, ignore_index=True) if resolved_rows else pd.DataFrame(columns=predictions_df.columns)


def compute_voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    VOC all-point AP interpolation.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_oe_score(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    iou_thresh: float = 0.5
) -> Dict[str, float]:
    """
    Calculates the full suite of Anveshak competition metrics:
      - mAP_K: Mean AP across all known classes
      - U-Rec, U-Prec, U-F1: Unknown class detection metrics
      - OE-Score: Combined harmonic metric
    """
    # 1. Apply conflict resolution first
    preds = resolve_conflicts(predictions_df, iou_thresh=iou_thresh)

    # 2. Extract known and unknown classes
    all_gt_classes = set(ground_truth_df['class_name'].unique())
    known_classes = sorted(list(all_gt_classes - {'unknown'}))

    # Evaluate AP for each known class
    aps = []
    for cls in known_classes:
        cls_gt = ground_truth_df[ground_truth_df['class_name'] == cls]
        cls_preds = preds[preds['class_name'] == cls].sort_values(by='score', ascending=False)

        n_gt = len(cls_gt)
        if n_gt == 0:
            continue

        n_preds = len(cls_preds)
        if n_preds == 0:
            aps.append(0.0)
            continue

        # Match predictions
        tp = np.zeros(n_preds)
        fp = np.zeros(n_preds)
        gt_matched = {img_id: set() for img_id in cls_gt['image_id'].unique()}

        # Note: Rule states: Known-class predictions that fall on unknown objects count as false positives.
        unknown_gt = ground_truth_df[ground_truth_df['class_name'] == 'unknown']

        for i, pred_row in enumerate(cls_preds.itertuples()):
            p_box = np.array([pred_row.xmin, pred_row.ymin, pred_row.xmax, pred_row.ymax])
            img_gts = cls_gt[cls_gt['image_id'] == pred_row.image_id]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_row in img_gts.iterrows():
                gt_box = np.array([gt_row.xmin, gt_row.ymin, gt_row.xmax, gt_row.ymax])
                iou = calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh and best_gt_idx not in gt_matched[pred_row.image_id]:
                tp[i] = 1.0
                gt_matched[pred_row.image_id].add(best_gt_idx)
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp)
        ap = compute_voc_ap(recalls, precisions)
        aps.append(ap)

    map_k = float(np.mean(aps)) if aps else 0.0

    # 3. Evaluate Unknown metrics (U-Rec, U-Prec, U-F1)
    unk_gt = ground_truth_df[ground_truth_df['class_name'] == 'unknown']
    unk_preds = preds[preds['class_name'] == 'unknown'].sort_values(by='score', ascending=False)

    total_unk_gt = len(unk_gt)
    total_unk_preds = len(unk_preds)

    unk_matched_gt = set()
    unk_tp = 0

    for pred_row in unk_preds.itertuples():
        p_box = np.array([pred_row.xmin, pred_row.ymin, pred_row.xmax, pred_row.ymax])
        img_unks = unk_gt[unk_gt['image_id'] == pred_row.image_id]

        best_iou = 0.0
        best_idx = -1
        for idx, gt_row in img_unks.iterrows():
            gt_box = np.array([gt_row.xmin, gt_row.ymin, gt_row.xmax, gt_row.ymax])
            iou = calculate_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_thresh and best_idx not in unk_matched_gt:
            unk_tp += 1
            unk_matched_gt.add(best_idx)

    u_rec = (unk_tp / total_unk_gt) if total_unk_gt > 0 else 0.0
    u_prec = (unk_tp / total_unk_preds) if total_unk_preds > 0 else 0.0
    u_f1 = (2 * u_rec * u_prec / (u_rec + u_prec)) if (u_rec + u_prec) > 0 else 0.0

    # 4. OE-Score Calculation
    if (map_k + u_f1) > 0:
        oe_score = 100.0 * 2.0 * (map_k * u_f1) / (map_k + u_f1)
    else:
        oe_score = 0.0

    return {
        "mAP_K": round(map_k, 4),
        "U-Rec": round(u_rec, 4),
        "U-Prec": round(u_prec, 4),
        "U-F1": round(u_f1, 4),
        "OE-Score": round(oe_score, 2)
    }
