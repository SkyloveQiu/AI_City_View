"""pipeline.stage4_5_intelligent_fmb

阶段4.5: 智能FMB优化
功能: 基于语义规则优化前景/中景/背景分层
作者: 基于Kai的智能FMB系统改编
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage
import cv2


class ForcedSemanticRules:
    """
    强制语义规则系统
    
    定义某些语义类别应该总是被分配到特定层
    例如: 天空总是背景
    """
    
    def __init__(self, semantic_config: List[Dict[str, Any]]):
        """
        Args:
            semantic_config: 语义配置列表，每个元素包含name, openness等字段
        """
        self.semantic_config = semantic_config
        self.forced_background_classes = []
        self.forced_foreground_classes = []
        self.forced_middleground_classes = []
        
        self._initialize_forced_rules()
    
    def _initialize_forced_rules(self):
        """根据类别名称初始化强制规则"""
        # 静默模式：不输出详细信息
        
        for idx, config_item in enumerate(self.semantic_config):
            name = config_item['name'].lower()
            class_id = idx  # ADE20K: class_id从0开始
            
            # 天空总是背景 - 强制规则（精确匹配，避免匹配skyscraper）
            if name == 'sky':
                self.forced_background_classes.append(class_id)
            
            # 海洋/大海总是背景（精确匹配）
            elif name == 'sea':
                self.forced_background_classes.append(class_id)
            
            # 地面/道路倾向于前景或中景（不强制）
            elif any(word in name for word in ['floor', 'flooring', 'ground', 'road', 'route']):
                # 不强制，但可以作为提示
                pass
    
    def get_forced_layer(self, semantic_class: int) -> Optional[int]:
        """
        获取语义类别的强制层
        
        Args:
            semantic_class: 语义类别ID (0-based for ADE20K)
        
        Returns:
            0: 前景, 1: 中景, 2: 背景, None: 无强制规则
        """
        if semantic_class in self.forced_background_classes:
            return 2  # 背景
        elif semantic_class in self.forced_foreground_classes:
            return 0  # 前景
        elif semantic_class in self.forced_middleground_classes:
            return 1  # 中景
        return None
    
    def is_forced(self, semantic_class: int) -> bool:
        """检查语义类别是否有强制规则"""
        return self.get_forced_layer(semantic_class) is not None


class IntelligentHoleFilling:
    """智能空洞填充系统"""
    
    min_hole_size = 10
    max_hole_size = 5000
    depth_threshold_ratio = 0.15
    
    def __init__(self, depth_map: np.ndarray, fmb_map: np.ndarray):
        """
        Args:
            depth_map: 深度图 (H, W) uint8, 值越大越远
            fmb_map: FMB分层图 (H, W), 0=前景, 1=中景, 2=背景
        """
        self.depth_map = depth_map.astype(np.float32)
        self.fmb_map = fmb_map
        self.H, self.W = depth_map.shape
        self.neighbor_radius = 5
    
    def process(self) -> Tuple[np.ndarray, Dict]:
        """
        处理并填充空洞
        
        Returns:
            filled_map: 填充后的FMB图
            fill_info: 填充统计信息
        """
        filled_map = self.fmb_map.copy()
        fill_info = {
            'total_holes_detected': 0,
            'holes_filled': 0,
            'holes_preserved': 0,
        }
        
        # 静默处理，不输出详细信息
        
        for layer in [0, 1, 2]:
            holes = self._detect_holes_in_layer(filled_map, layer)
            fill_info['total_holes_detected'] += len(holes)
            
            for hole_mask in holes:
                hole_size = np.sum(hole_mask)
                
                if hole_size < self.min_hole_size or hole_size > self.max_hole_size:
                    continue
                
                should_fill, analysis = self._analyze_hole(hole_mask, layer)
                
                if should_fill:
                    filled_map[hole_mask] = layer
                    fill_info['holes_filled'] += 1
                else:
                    fill_info['holes_preserved'] += 1
        
        return filled_map, fill_info
    
    def _detect_holes_in_layer(self, fmb_map: np.ndarray, layer: int) -> List[np.ndarray]:
        """检测某一层内的空洞"""
        layer_mask = (fmb_map == layer).astype(np.uint8)
        
        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        layer_closed = cv2.morphologyEx(layer_mask, cv2.MORPH_CLOSE, kernel)
        
        # 填充孔洞
        layer_filled = self._fill_holes_morphological(layer_closed)
        
        # 空洞 = 填充后的区域 - 原始区域
        holes_mask = layer_filled & (~layer_mask.astype(bool))
        
        # 标记连通区域
        labeled_holes, num_holes = ndimage.label(holes_mask)
        
        holes = []
        for hole_id in range(1, num_holes + 1):
            hole_mask = labeled_holes == hole_id
            if self._is_hole_surrounded(hole_mask, fmb_map, layer):
                holes.append(hole_mask)
        
        return holes
    
    def _fill_holes_morphological(self, binary_mask: np.ndarray) -> np.ndarray:
        """使用形态学方法填充空洞"""
        filled = binary_mask.copy()
        h, w = binary_mask.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, mask, (0, 0), 1)
        filled_inv = cv2.bitwise_not(filled)
        result = binary_mask | filled_inv
        return result.astype(np.uint8)
    
    def _is_hole_surrounded(self, hole_mask: np.ndarray, fmb_map: np.ndarray, layer: int) -> bool:
        """检查空洞是否被目标层包围"""
        dilated = ndimage.binary_dilation(hole_mask, iterations=2)
        boundary = dilated & (~hole_mask)
        boundary_values = fmb_map[boundary]
        
        if len(boundary_values) == 0:
            return False
        
        unique_values, counts = np.unique(boundary_values, return_counts=True)
        target_ratio = 0.0
        
        for val, count in zip(unique_values, counts):
            if val == layer:
                target_ratio = count / np.sum(counts)
                break
        
        return target_ratio > 0.8
    
    def _analyze_hole(self, hole_mask: np.ndarray, surrounding_layer: int) -> Tuple[bool, Dict]:
        """分析空洞是否应该填充"""
        hole_depths = self.depth_map[hole_mask]
        hole_mean_depth = np.mean(hole_depths)
        hole_std_depth = np.std(hole_depths)
        
        # 获取周围邻域
        dilated = ndimage.binary_dilation(hole_mask, iterations=self.neighbor_radius)
        neighbor_mask = dilated & (~hole_mask) & (self.fmb_map == surrounding_layer)
        
        if np.sum(neighbor_mask) == 0:
            return False, {'decision_reason': 'no_valid_neighbors'}
        
        neighbor_depths = self.depth_map[neighbor_mask]
        neighbor_mean_depth = np.mean(neighbor_depths)
        neighbor_std_depth = np.std(neighbor_depths)
        
        # 计算深度差异
        depth_difference = abs(hole_mean_depth - neighbor_mean_depth)
        
        # 归一化深度差异
        local_depths = np.concatenate([hole_depths, neighbor_depths])
        local_depth_range = np.max(local_depths) - np.min(local_depths)
        if local_depth_range < 1:
            local_depth_range = 1
        
        normalized_difference = depth_difference / local_depth_range
        
        # 决策
        should_fill = normalized_difference < self.depth_threshold_ratio
        
        # 如果空洞内部方差过大，不填充
        if hole_std_depth > neighbor_std_depth * 2:
            should_fill = False
        
        analysis = {
            'depth_difference': depth_difference,
            'normalized_difference': normalized_difference,
            'decision_reason': 'small_depth_difference' if should_fill else 'large_depth_difference'
        }
        
        return should_fill, analysis


def stage4_5_intelligent_fmb(
    depth_map: np.ndarray,
    semantic_map: np.ndarray,
    foreground_mask: np.ndarray,
    middleground_mask: np.ndarray,
    background_mask: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    阶段4.5: 智能FMB优化
    
    应用强制语义规则和智能空洞填充
    
    Args:
        depth_map: 深度图 (H, W) uint8
        semantic_map: 语义分割图 (H, W) uint8, class_id从0开始
        foreground_mask: 前景mask (H, W) bool
        middleground_mask: 中景mask (H, W) bool
        background_mask: 背景mask (H, W) bool
        config: 配置字典，需包含semantic_config
    
    Returns:
        包含优化后的mask和统计信息的字典
    """
    if depth_map.ndim != 2 or depth_map.dtype != np.uint8:
        raise ValueError(f"depth_map必须是(H,W) uint8，当前shape={depth_map.shape}, dtype={depth_map.dtype}")
    if semantic_map.ndim != 2 or semantic_map.dtype != np.uint8:
        raise ValueError(f"semantic_map必须是(H,W) uint8")
    
    H, W = depth_map.shape
    
    # 获取语义配置
    semantic_config = config.get('semantic_config', [])
    if not semantic_config:
        print("  [智能FMB] 警告: 未找到semantic_config，跳过智能优化")
        return {
            'foreground_mask': foreground_mask,
            'middleground_mask': middleground_mask,
            'background_mask': background_mask,
            'optimization_applied': False,
        }
    
    # 初始化强制规则
    forced_rules = ForcedSemanticRules(semantic_config)
    
    # 构建FMB图 (0=前景, 1=中景, 2=背景)
    fmb_map = np.zeros((H, W), dtype=np.uint8)
    fmb_map[foreground_mask] = 0
    fmb_map[middleground_mask] = 1
    fmb_map[background_mask] = 2
    
    # 应用强制语义规则（静默模式）
    rules_applied = 0
    pixels_changed_by_rules = 0
    
    for class_id in range(len(semantic_config)):
        forced_layer = forced_rules.get_forced_layer(class_id)
        if forced_layer is not None:
            class_mask = semantic_map == class_id
            pixel_count = np.sum(class_mask)
            if pixel_count > 0:
                fmb_map[class_mask] = forced_layer
                rules_applied += 1
                pixels_changed_by_rules += pixel_count
    
    # 智能空洞填充
    enable_hole_filling = config.get('enable_intelligent_hole_filling', True)
    fill_info = {}
    
    if enable_hole_filling:
        hole_filler = IntelligentHoleFilling(depth_map, fmb_map)
        fmb_map, fill_info = hole_filler.process()
    
    # 转换回mask
    optimized_foreground_mask = (fmb_map == 0)
    optimized_middleground_mask = (fmb_map == 1)
    optimized_background_mask = (fmb_map == 2)
    
    # 统计优化效果（静默模式）
    fg_changed = np.sum(optimized_foreground_mask != foreground_mask)
    mg_changed = np.sum(optimized_middleground_mask != middleground_mask)
    bg_changed = np.sum(optimized_background_mask != background_mask)
    total_changed = fg_changed + mg_changed + bg_changed
    
    return {
        'foreground_mask': optimized_foreground_mask.astype(bool),
        'middleground_mask': optimized_middleground_mask.astype(bool),
        'background_mask': optimized_background_mask.astype(bool),
        'optimization_applied': True,
        'rules_applied': rules_applied,
        'pixels_changed': int(total_changed),
        'fill_info': fill_info,
    }
