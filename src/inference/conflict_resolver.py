"""
Inference & Conflict Resolution Script for Anveshak
Takes raw detector predictions and enforces:
1. Coordinates formatting (integers within image bounds, xmin < xmax, ymin < ymax)
2. Score formatting (float 0.0 to 1.0)
3. Maximum 1,000 predictions per image
4. Conflict Rule resolution (IoU >= 0.5 between known and unknown: higher score wins, ties go to earlier row)
5. Output CSV structure: image_id,class_name,score,xmin,ymin,xmax,ymax
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils.metrics import resolve_conflicts


def clean_and_format_predictions(
    df: pd.DataFrame,
    image_width: int = 512,
    image_height: int = 512,
    max_per_image: int = 1000
) -> pd.DataFrame:
    """
    Cleans raw predictions to conform strictly to validate_predictions.py requirements.
    """
    if df.empty:
        return df

    # 1. Clip bounding boxes to valid tile coordinates
    df['xmin'] = df['xmin'].clip(lower=0, upper=image_width - 1).astype(int)
    df['ymin'] = df['ymin'].clip(lower=0, upper=image_height - 1).astype(int)
    df['xmax'] = df['xmax'].clip(lower=1, upper=image_width).astype(int)
    df['ymax'] = df['ymax'].clip(lower=1, upper=image_height).astype(int)

    # 2. Filter invalid degenerated boxes
    valid_box = (df['xmin'] < df['xmax']) & (df['ymin'] < df['ymax'])
    df = df[valid_box].copy()

    # 3. Ensure score is between 0 and 1 rounded to 4 decimals
    df['score'] = df['score'].clip(lower=0.0, upper=1.0).round(4)

    # 4. Remove duplicate rows if any
    df = df.drop_duplicates(subset=['image_id', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax']).copy()

    # 5. Resolve IoU >= 0.5 conflicts between known and unknown predictions
    df = resolve_conflicts(df, iou_thresh=0.5)

    # 6. Keep top max_per_image sorted by score per image
    df = (
        df.sort_values(by=['image_id', 'score'], ascending=[True, False])
        .groupby('image_id')
        .head(max_per_image)
        .reset_index(drop=True)
    )

    # 7. Select standard columns
    cols = ['image_id', 'class_name', 'score', 'xmin', 'ymin', 'xmax', 'ymax']
    return df[cols]


def save_submission_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves predictions as a single UTF-8 CSV file without row index.
    """
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[✓] Saved {len(df)} predictions to {output_path}")
