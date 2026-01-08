"""pipeline.stage5_openness

阶段5: 开放度计算
功能: 基于语义类别计算开放度图
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def stage5_openness(semantic_map: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """阶段5: 开放度计算。

    - openness_config: 与 classes 等长，元素为 0/1
    - openness_map: 0(封闭) 或 255(开放)
    """
    if semantic_map.ndim != 2:
        raise ValueError(f"semantic_map 必须是二维 (H, W)，当前 shape={semantic_map.shape}")
    if semantic_map.dtype != np.uint8:
        raise ValueError(f"semantic_map 必须是 uint8，当前 dtype={semantic_map.dtype}")

    classes: List[str] = list(config.get('classes', []) or [])
    openness_config: List[int] = list(config.get('openness_config', []) or [])

    # 允许 ADE20K 等“多类语义图”场景：openness_config/classes 可以只配置一部分。
    # 未指定的 class_id 默认视为封闭(0)。
    max_id = int(semantic_map.max())
    needed_len = max(0, max_id + 1)  # 语义图可能包含 0..max_id

    # Common convention:
    # - LangSAM produces ids: 0=background, 1..N correspond to `classes` prompts
    # In that case, openness_config is usually length N+1, while classes is length N.
    # For reporting, build a name list aligned to ids.
    names_by_id: List[str]
    if len(openness_config) == len(classes) + 1:
        names_by_id = ['background'] + classes
    else:
        names_by_id = list(classes)
    if len(openness_config) < needed_len:
        openness_config = openness_config + [0] * (needed_len - len(openness_config))
    if len(names_by_id) < needed_len:
        # names_by_id 仅用于统计展示；不足时用占位名补齐
        names_by_id = names_by_id + [f"class_{i}" for i in range(len(names_by_id), needed_len)]

    for idx, value in enumerate(openness_config):
        if int(value) not in (0, 1):
            raise ValueError(f"openness_config[{idx}] 必须是0或1，当前={value}")

    H, W = semantic_map.shape
    total_pixels = int(H * W)
    openness_map = np.zeros((H, W), dtype=np.uint8)

    # 逐类映射：支持 0..max_id
    by_class: Dict[str, Any] = {}
    for class_id in range(0, max_id + 1):
        class_pixels = semantic_map == class_id
        class_count = int(class_pixels.sum())
        is_open = int(openness_config[class_id])

        if is_open == 1:
            openness_map[class_pixels] = 255

        if class_count > 0:
            by_class[names_by_id[class_id]] = {
                'pixels': class_count,
                'is_open': bool(is_open),
                'contribution_to_openness': float(class_count / total_pixels) if total_pixels else 0.0,
            }

    open_pixels = int(np.sum(openness_map == 255))
    closed_pixels = int(np.sum(openness_map == 0))
    openness_ratio = float(open_pixels / total_pixels) if total_pixels else 0.0

    return {
        'openness_map': openness_map,
        'openness_stats': {
            'open_pixels': open_pixels,
            'closed_pixels': closed_pixels,
            'total_pixels': total_pixels,
            'openness_ratio': openness_ratio,
            'openness_percent': float(openness_ratio * 100.0),
            'by_class': by_class,
        },
    }


