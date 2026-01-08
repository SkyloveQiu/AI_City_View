"""pipeline.stage4_depth_layering

阶段4: 景深分层
功能: 将深度图分为前景/中景/背景三层
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def stage4_depth_layering(depth_map: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """阶段4: 景深分层。

    depth_map: (H, W) uint8, 值范围 [0, 255]
      - 0: 最近（前景）
      - 255: 最远（背景）
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map 必须是二维 (H, W)，当前 shape={depth_map.shape}")
    if depth_map.dtype != np.uint8:
        raise ValueError(f"depth_map 必须是 uint8，当前 dtype={depth_map.dtype}")

    split_method = str(config.get('split_method', 'percentile')).lower()
    fg_ratio = float(config.get('fg_ratio', 0.33))
    bg_ratio = float(config.get('bg_ratio', 0.33))

    H, W = depth_map.shape
    total_pixels = int(H * W)

    if split_method == 'fixed_threshold':
        threshold_1 = float(config.get('threshold_1', 85))
        threshold_2 = float(config.get('threshold_2', 170))
        t1, t2 = threshold_1, threshold_2
    else:
        # 默认：百分位数
        # P1 为前景比例，P2 为 1 - 背景比例
        fg_ratio = min(max(fg_ratio, 0.0), 1.0)
        bg_ratio = min(max(bg_ratio, 0.0), 1.0)
        p1 = fg_ratio * 100.0
        p2 = (1.0 - bg_ratio) * 100.0
        if p1 >= p2:
            # 回退到经典 33/66
            p1, p2 = 33.0, 66.0

        t1 = float(np.percentile(depth_map, p1))
        t2 = float(np.percentile(depth_map, p2))

    # 掩码（互斥且覆盖所有像素）
    foreground_mask = depth_map <= t1
    background_mask = depth_map > t2
    middleground_mask = ~(foreground_mask | background_mask)

    fg_pixels = int(foreground_mask.sum())
    mg_pixels = int(middleground_mask.sum())
    bg_pixels = int(background_mask.sum())

    # 完整性修正：理论上应相加等于 total_pixels
    if fg_pixels + mg_pixels + bg_pixels != total_pixels:
        # 极小概率由 dtype/广播导致，强制修正
        middleground_mask = ~(foreground_mask | background_mask)
        fg_pixels = int(foreground_mask.sum())
        mg_pixels = int(middleground_mask.sum())
        bg_pixels = int(background_mask.sum())

    def _percent(x: int) -> float:
        return (x / total_pixels * 100.0) if total_pixels > 0 else 0.0

    return {
        'foreground_mask': foreground_mask.astype(bool),
        'middleground_mask': middleground_mask.astype(bool),
        'background_mask': background_mask.astype(bool),
        'depth_thresholds': {
            'P33': float(t1),
            'P66': float(t2),
        },
        'layer_stats': {
            'foreground_pixels': fg_pixels,
            'middleground_pixels': mg_pixels,
            'background_pixels': bg_pixels,
            'foreground_percent': float(_percent(fg_pixels)),
            'middleground_percent': float(_percent(mg_pixels)),
            'background_percent': float(_percent(bg_pixels)),
        },
    }


