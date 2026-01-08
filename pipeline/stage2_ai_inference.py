"""
阶段2: AI模型推理
功能: 语义分割 (SAM 2.1 + LangSAM) + 深度估计 (Depth Anything V2)
精度优先方案
"""

from __future__ import annotations

import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import threading

# 延迟导入，避免未安装时直接报错
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，AI推理功能将不可用")


_LANGSAM_PREDICT_LOCK = threading.Lock()
_ONEFORMER_LOCK = threading.Lock()


def stage2_ai_inference(image: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    阶段2: AI模型推理
    
    参数:
        image: np.ndarray - 来自阶段1的 original (H, W, 3) BGR格式
        config: dict - 配置参数
            - classes: List[str] - 语义类别列表
            - encoder: str - 模型大小 ('vitb' 或 'vits')
            - class_colors: Dict[int, List[int]] - 类别颜色映射
    
    返回:
        dict - 包含以下键:
            - semantic_map: np.ndarray - 语义分割图 (H, W) uint8
            - depth_map: np.ndarray - 深度图 (H, W) uint8
    """
    H, W = image.shape[:2]
    
    # 语义分割 (SAM 2.1 + LangSAM)
    semantic_map = _semantic_segmentation(image, config)
    
    # 深度估计 (Depth Anything V2)
    depth_map = _depth_estimation(image, config)
    
    return {
        'semantic_map': semantic_map,
        'depth_map': depth_map
    }


def _semantic_segmentation(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    语义分割 - 使用 SAM 2.1 + LangSAM
    
    参数:
        image: (H, W, 3) BGR uint8
        config: dict - 配置参数
    
    返回:
        semantic_map: (H, W) uint8, 值范围 [0, N]
            - 0: 背景/未分类
            - 1-N: 语义类别ID (N = len(classes))
    """
    H, W = image.shape[:2]
    backend = str(config.get('semantic_backend', 'oneformer_ade20k')).strip().lower()

    # 初始化语义分割图
    semantic_map = np.zeros((H, W), dtype=np.uint8)

    if backend.startswith('oneformer'):
        try:
            profile = bool(config.get('profile', False))
            t0 = time.perf_counter()

            model, processor = get_semantic_model(config)
            if model is None or processor is None:
                print("  ⚠️  警告: OneFormer 未就绪，语义分割使用占位实现")
                return semantic_map

            _maybe_apply_semantic_items_mapping_for_ade20k(model, config)

            if profile:
                print(f"  ⏱️  OneFormer ready: {time.perf_counter() - t0:.3f}s")

            device = _get_torch_device(config)
            use_fp16 = bool(config.get('semantic_use_fp16', True))

            # OpenCV(BGR)->RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # OneFormer semantic segmentation
            t1 = time.perf_counter()
            with torch.inference_mode():
                inputs = processor(images=rgb, task_inputs=["semantic"], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if use_fp16 and device.type == 'cuda':
                    inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
                outputs = model(**inputs)
                pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
            if profile:
                print(f"  ⏱️  OneFormer inference+post: {time.perf_counter() - t1:.3f}s")

            semantic_map = pred.detach().to('cpu').numpy().astype(np.uint8)
            return semantic_map

        except Exception as e:
            print(f"  ❌ OneFormer 语义分割出错: {e}")
            import traceback
            traceback.print_exc()
            return semantic_map

    # 兼容旧实现：LangSAM 文本提示分割
    classes = list(config.get('classes', []) or [])
    semantic_cfg = config.get('semantic', {}) or {}

    if len(classes) == 0:
        print("  警告: 未指定类别，返回全0语义图")
        return semantic_map

    try:
        profile = bool(config.get('profile', False))
        t0 = time.perf_counter()

        model, _processor = get_semantic_model({**config, 'semantic_backend': 'langsam'})
        if model is None:
            print("  ⚠️  警告: LangSAM 未就绪，语义分割使用占位实现")
            _generate_placeholder_semantic_map(semantic_map, classes, H, W)
            return semantic_map

        if profile:
            print(f"  ⏱️  LangSAM ready: {time.perf_counter() - t0:.3f}s")

        # OpenCV(BGR)->RGB PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        for class_id, class_name in enumerate(classes, start=1):
            if profile:
                t_cls = time.perf_counter()
            masks = _langsam_predict_masks(model, pil_img, class_name, semantic_cfg)

            if masks is None:
                continue
            if masks.ndim == 2:
                combined = masks.astype(bool)
            else:
                combined = np.any(masks.astype(bool), axis=0)

            if combined.any():
                semantic_map[combined] = class_id

            if profile:
                print(f"  ⏱️  semantic '{class_name}': {time.perf_counter() - t_cls:.3f}s")

    except Exception as e:
        print(f"  ❌ LangSAM 语义分割出错: {e}")
        import traceback
        traceback.print_exc()
        return semantic_map

    return semantic_map


def _normalize_label(s: str) -> str:
    import re

    s = s.lower().strip()
    # Normalize separators and punctuation.
    s = s.replace('&', ' and ')
    s = re.sub(r"[\[\]\(\)\{\}]+", " ", s)
    s = re.sub(r"[;,:/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _maybe_apply_semantic_items_mapping_for_ade20k(model, config: Dict[str, Any]) -> None:
    """Build colors/openness for OneFormer ADE20K ids from user-provided semantic_items.

    - OneFormer outputs class ids 0..149 (ADE20K-150)
    - User semantic config contains a small set of human labels with colors/openness.
    This function tries to match those labels to ADE20K id2label and fills:
      - config['classes'] as id-aligned label list (for stats)
      - config['colors'] mapping for matched ids (others handled by stage6 fallback)
      - config['openness_config'] list length max_id+1 with matched ids set
    """
    if config.get('_ade20k_mapped_from_semantic_items'):
        return
    backend = str(config.get('semantic_backend', '')).strip().lower()
    if not backend.startswith('oneformer'):
        return

    items = config.get('semantic_items', None)
    if not items:
        return

    id2label = getattr(getattr(model, 'config', None), 'id2label', None)
    if not isinstance(id2label, dict) or not id2label:
        return

    # Build id-aligned label list
    max_id = max(int(k) for k in id2label.keys() if str(k).isdigit()) if any(str(k).isdigit() for k in id2label.keys()) else 149
    labels_by_id = []
    for i in range(0, max_id + 1):
        labels_by_id.append(str(id2label.get(i, f"class_{i}")))

    # Build normalized label->id index
    norm_to_id = {}
    for i, lab in enumerate(labels_by_id):
        norm_to_id[_normalize_label(lab)] = i

    colors = {0: (0, 0, 0)}
    openness_config = [0] * (max_id + 1)

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name', '')).strip()
        if not name:
            continue
        openness = int(item.get('openness', 0) or 0)
        bgr = item.get('bgr', None)
        if bgr is None:
            continue

        # Try exact normalized match; then try synonyms split by ';'
        candidates = [name] + [p.strip() for p in name.split(';') if p.strip()]
        matched_id = None
        for cand in candidates:
            nid = norm_to_id.get(_normalize_label(cand), None)
            if nid is not None:
                matched_id = int(nid)
                break

        # Fallback: substring match (avoid ambiguous matches)
        if matched_id is None:
            n = _normalize_label(name)
            hits = [i for k, i in norm_to_id.items() if (n == k or n in k or k in n)]
            if len(hits) == 1:
                matched_id = int(hits[0])

        if matched_id is None or matched_id <= 0 or matched_id > max_id:
            continue

        colors[matched_id] = tuple(int(x) for x in bgr)
        openness_config[matched_id] = 1 if openness == 1 else 0

    config['classes'] = labels_by_id
    config['colors'] = colors
    config['openness_config'] = openness_config
    config['_ade20k_mapped_from_semantic_items'] = True


def _depth_estimation(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    深度估计 - 使用 Depth Anything V2
    
    参数:
        image: (H, W, 3) BGR uint8
        config: dict - 配置参数
    
    返回:
        depth_map: (H, W) uint8, 值范围 [0, 255]
            - 0: 最近（前景）
            - 255: 最远（背景）
    """
    H, W = image.shape[:2]
    
    try:
        profile = bool(config.get('profile', False))
        t0 = time.perf_counter()

        depth_model, depth_processor = get_depth_model(config)
        if depth_model is None or depth_processor is None:
            print(f"  ⚠️  警告: Depth Anything V2 未就绪，使用占位深度图")
            return _generate_placeholder_depth_map(H, W)

        if profile:
            print(f"  ⏱️  Depth model ready: {time.perf_counter() - t0:.3f}s")

        device = _get_torch_device(config)
        use_fp16 = bool(config.get('depth_use_fp16', True))

        # OpenCV(BGR)->RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推理
        t1 = time.perf_counter()
        with torch.inference_mode():
            inputs = depth_processor(images=rgb, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if use_fp16 and device.type == 'cuda':
                inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

            outputs = depth_model(**inputs)

            # transformers 的深度模型一般有 predicted_depth
            if hasattr(outputs, 'predicted_depth'):
                pred = outputs.predicted_depth
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                pred = outputs[0]
            else:
                pred = getattr(outputs, 'logits', None)

            if pred is None:
                raise RuntimeError("Depth Anything V2 输出不包含 predicted_depth/logits")

            # pred: (B, H', W')
            pred = pred.squeeze(0)
            pred = pred.detach().float().cpu().numpy()

        if profile:
            print(f"  ⏱️  Depth inference: {time.perf_counter() - t1:.3f}s")

        # 将预测结果 resize 回原图大小
        t2 = time.perf_counter()
        pred_resized = cv2.resize(pred, (W, H), interpolation=cv2.INTER_CUBIC)

        depth_map = _normalize_depth_to_uint8(pred_resized, invert=bool(config.get('depth_invert', True)))

        if profile:
            print(f"  ⏱️  Depth postprocess (resize+norm): {time.perf_counter() - t2:.3f}s")
        
    except ImportError as e:
        print(f"  ❌ 错误: 缺少必要的库，无法执行深度估计")
        print(f"  请安装: pip install transformers accelerate")
        print(f"  错误详情: {e}")
        # 返回简单梯度图，不中断流程
        return _generate_placeholder_depth_map(H, W)
    
    except Exception as e:
        print(f"  ❌ 深度估计出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回简单梯度图，不中断流程
        return _generate_placeholder_depth_map(H, W)
    
    return depth_map


def _normalize_depth_to_uint8(depth_raw: np.ndarray, invert: bool = True) -> np.ndarray:
    """把 float 深度/逆深度图归一化到 uint8 [0,255]。

    约定输出：0=近，255=远。
    Depth Anything V2 的输出通常更像“逆深度”（近处更大），因此默认 invert=True。
    """
    depth_min = float(np.min(depth_raw))
    depth_max = float(np.max(depth_raw))
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max == depth_min:
        return np.full(depth_raw.shape, 128, dtype=np.uint8)

    norm = (depth_raw - depth_min) / (depth_max - depth_min)
    norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    if invert:
        norm = (255 - norm).astype(np.uint8)
    return norm


def _generate_placeholder_semantic_map(semantic_map: np.ndarray, classes: List[str], H: int, W: int) -> None:
    """
    生成占位语义图 (仅用于测试)
    简单的垂直分块模式
    """
    num_classes = len(classes)
    if num_classes == 0:
        return
    
    # 简单的垂直分块
    block_height = H // num_classes
    for i, class_id in enumerate(range(1, num_classes + 1)):
        y_start = i * block_height
        y_end = (i + 1) * block_height if i < num_classes - 1 else H
        semantic_map[y_start:y_end, :] = class_id


def _generate_placeholder_depth_map(H: int, W: int) -> np.ndarray:
    """
    生成占位深度图 (仅用于测试)
    简单的垂直梯度
    """
    depth_map = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        depth_map[y, :] = int((y / H) * 255)
    return depth_map


# ============================================
# 模型缓存管理 (用于优化性能)
# ============================================

_model_cache = {
    'semantic_model': None,
    'semantic_processor': None,
    'semantic_backend': None,
    'semantic_device': None,
    'depth_model': None,
    'depth_processor': None,
}


def get_semantic_model(config: Dict[str, Any]):
    """
    获取语义分割模型 (带缓存)
    只初始化一次，后续复用
    """
    if not TORCH_AVAILABLE:
        return None, None

    backend = str(config.get('semantic_backend', 'oneformer_ade20k')).strip().lower()
    device = _get_torch_device(config)

    # If backend changes between calls, reset semantic cache.
    if _model_cache.get('semantic_backend') != backend:
        _model_cache['semantic_model'] = None
        _model_cache['semantic_processor'] = None
        _model_cache['semantic_backend'] = backend
        _model_cache['semantic_device'] = None

    if _model_cache['semantic_model'] is not None:
        return _model_cache['semantic_model'], _model_cache['semantic_processor']

    if backend.startswith('oneformer'):
        try:
            profile = bool(config.get('profile', False))
            t_import = time.perf_counter()
            print("  初始化 OneFormer（首次导入/下载权重可能较慢）...")
            from transformers import AutoProcessor, OneFormerForUniversalSegmentation
            if profile:
                print(f"  ⏱️  import transformers(oneformer): {time.perf_counter() - t_import:.3f}s")
        except Exception as e:
            print(f"  ❌ 无法导入 OneFormer 相关依赖: {e}")
            return None, None

        model_id = str(config.get('oneformer_model_id', 'shi-labs/oneformer_ade20k_swin_large'))
        use_fp16 = bool(config.get('semantic_use_fp16', True))
        print(f"  加载 OneFormer(ADE20K-150): {model_id} (device={device}, fp16={use_fp16})")

        t_load = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_id)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id)
        model.eval()

        if device.type == 'cuda':
            model.to(device)
            if use_fp16:
                model.half()

        if bool(config.get('profile', False)):
            print(f"  ⏱️  load processor+model: {time.perf_counter() - t_load:.3f}s")

        _model_cache['semantic_model'] = model
        _model_cache['semantic_processor'] = processor
        _model_cache['semantic_backend'] = backend
        _model_cache['semantic_device'] = str(device)
        return model, processor

    # LangSAM backend
    try:
        profile = bool(config.get('profile', False))
        t_import = time.perf_counter()
        print("  初始化 LangSAM（首次导入/下载权重可能较慢）...")
        from lang_sam import LangSAM
        if profile:
            print(f"  ⏱️  import lang_sam: {time.perf_counter() - t_import:.3f}s")
    except Exception as e:
        print(f"  ❌ 无法导入 lang_sam: {e}")
        return None, None

    # LangSAM (v0.2.x) 基于 SAM2.1，支持的 key：
    # sam2.1_hiera_tiny / sam2.1_hiera_small / sam2.1_hiera_base_plus / sam2.1_hiera_large
    sam_type = str(config.get('sam_type', '')).strip()
    if not sam_type:
        encoder = str(config.get('encoder', 'vitb')).lower()
        if encoder in ('vitb', 'vit_b', 'b'):
            sam_type = 'sam2.1_hiera_base_plus'
        elif encoder in ('vits', 'vit_s', 's'):
            sam_type = 'sam2.1_hiera_small'
        else:
            sam_type = 'sam2.1_hiera_small'

    print(f"  加载 LangSAM (sam_type={sam_type}, device={device})")
    model = LangSAM(sam_type=sam_type, device=device)

    _model_cache['semantic_model'] = model
    _model_cache['semantic_processor'] = None
    _model_cache['semantic_backend'] = backend
    _model_cache['semantic_device'] = str(device)
    return model, None


def _langsam_predict_masks(model, pil_img: Image.Image, text_prompt: str, config: Dict[str, Any] | None = None) -> Optional[np.ndarray]:
    """调用 LangSAM 并返回 masks ndarray。

    兼容不同版本返回格式：可能是 (masks, boxes, phrases, logits) 或 dict。
    返回:
      - None: 没有检测到
      - np.ndarray: (N,H,W) 或 (H,W)
    """
    cfg = config or {}
    box_threshold = float(cfg.get('box_threshold', 0.3))
    text_threshold = float(cfg.get('text_threshold', 0.25))

    # LangSAM/SAM2 predictor is not thread-safe; serialize predict calls.
    with _LANGSAM_PREDICT_LOCK:
        # LangSAM.predict expects lists and returns list[dict] (one per image)
        results = model.predict([pil_img], [text_prompt], box_threshold=box_threshold, text_threshold=text_threshold)
    if not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None

    masks = first.get('masks', None)
    if masks is None:
        return None

    masks = np.asarray(masks)
    if masks.size == 0:
        return None
    return masks


def get_depth_model(config: Dict[str, Any]):
    """
    获取深度估计模型 (带缓存)
    只初始化一次，后续复用
    """
    if _model_cache['depth_model'] is None:
        if not TORCH_AVAILABLE:
            return None, None

        try:
            # 这一段在首次运行时可能会“卡住”，通常是 transformers/torch 的大导入 + CUDA 初始化，并非 bug。
            profile = bool(config.get('profile', False))
            t_import = time.perf_counter()
            print("  初始化 Depth Anything V2（导入 transformers，首次可能较慢）...")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            if profile:
                print(f"  ⏱️  import transformers: {time.perf_counter() - t_import:.3f}s")
        except Exception as e:
            print(f"  ❌ 无法导入 transformers 深度模型: {e}")
            return None, None

        model_id = str(config.get('depth_model_id', 'depth-anything/Depth-Anything-V2-Base-hf'))
        device = _get_torch_device(config)
        use_fp16 = bool(config.get('depth_use_fp16', True))

        print(f"  加载 Depth Anything V2: {model_id} (device={device}, fp16={use_fp16})")

        t_load = time.perf_counter()
        # transformers 会提示 future default use_fast=True；这里尽量显式指定。
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        except TypeError:
            processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        if profile:
            print(f"  ⏱️  load processor+model: {time.perf_counter() - t_load:.3f}s")
        model.eval()

        if device.type == 'cuda':
            t_to = time.perf_counter()
            model.to(device)
            if use_fp16:
                model.half()
            if profile:
                print(f"  ⏱️  model.to(device)/half: {time.perf_counter() - t_to:.3f}s")

        _model_cache['depth_model'] = model
        _model_cache['depth_processor'] = processor

    return _model_cache['depth_model'], _model_cache['depth_processor']


def _get_torch_device(config: Dict[str, Any]):
    if not TORCH_AVAILABLE:
        return None
    device_str = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    return torch.device(device_str)


def clear_model_cache():
    """
    清除模型缓存 (释放GPU内存)
    """
    global _model_cache
    _model_cache['semantic_model'] = None
    _model_cache['semantic_processor'] = None
    _model_cache['semantic_backend'] = None
    _model_cache['semantic_device'] = None
    _model_cache['depth_model'] = None
    _model_cache['depth_processor'] = None
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
