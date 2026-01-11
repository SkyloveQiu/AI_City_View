"""
视觉分析Pipeline - 单图处理主入口
处理1张全景图，自动分割为前后两部分，每部分生成21张输出图片
支持多线程批处理（GPU操作使用线程锁保护）
"""

import sys
import cv2
import time
import threading
from pathlib import Path
from typing import Any, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stage1_preprocess import crop_and_split_panorama
from pipeline.stage2_ai_inference import stage2_ai_inference
from pipeline.stage3_postprocess import stage3_postprocess
from pipeline.stage4_depth_layering import stage4_depth_layering
from pipeline.stage5_openness import stage5_openness
from pipeline.stage6_generate_images import stage6_generate_images
from pipeline.stage7_save_outputs import stage7_save_outputs

# GPU操作的全局锁（确保多线程时GPU安全）
_gpu_lock = threading.Lock()
from pipeline.stage7_save_outputs import stage7_save_outputs


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    import json
    config_path = Path(__file__).parent / "Semantic_configuration.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        semantic_config = json.load(f)
    
    return {
        'split_method': 'percentile',
        'semantic_config': semantic_config,
    }


def process_half_image(
    image_data: np.ndarray,
    basename: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    处理单个半图（前半部分或后半部分）
    """
    height, width = image_data.shape[:2]
    
    # Stage1数据（直接使用传入的图片数据）
    stage1_data = {
        'original': image_data,
        'original_copy': image_data.copy(),
        'height': height,
        'width': width,
        'metadata': {
            'basename': basename,
            'width': width,
            'height': height,
        }
    }
    
    # Stage 2: AI推理（使用GPU锁保护）
    print("  Stage 2: AI推理...")
    with _gpu_lock:
        stage2_result = stage2_ai_inference(stage1_data['original'], config)
    
    # Stage 3: 后处理
    print("  Stage 3: 后处理...")
    stage3_result = stage3_postprocess(
        stage2_result['semantic_map'],
        config
    )
    
    # 将depth_map添加到stage3_result中
    stage3_result['depth_map'] = stage2_result['depth_map']
    
    # Stage 4: 景深分层
    print("  Stage 4: 景深分层...")
    stage4_result = stage4_depth_layering(
        stage3_result['depth_map'],
        config
    )
    
    # Stage 5: 开放度计算
    print("  Stage 5: 开放度计算...")
    stage5_result = stage5_openness(
        stage3_result['semantic_map_processed'],
        config
    )
    
    # 将mask数据添加到stage5_result中
    stage5_result['foreground_mask'] = stage4_result['foreground_mask']
    stage5_result['middleground_mask'] = stage4_result['middleground_mask']
    stage5_result['background_mask'] = stage4_result['background_mask']
    
    # Stage 6: 生成图片
    print("  Stage 6: 生成图片...")
    stage6_result = stage6_generate_images(
        stage1_data['original_copy'],
        stage3_result['semantic_map_processed'],
        stage3_result['depth_map'],
        stage5_result['openness_map'],
        stage5_result['foreground_mask'],
        stage5_result['middleground_mask'],
        stage5_result['background_mask'],
        config
    )
    
    # Stage 7: 保存输出
    print("  Stage 7: 保存输出...")
    stage7_result = stage7_save_outputs(
        stage6_result['images'],
        output_dir,
        basename,
        stage1_data['metadata']
    )
    
    return {
        'success': True,
        'output_dir': output_dir,
        'saved_files': stage7_result.get('saved_files', [])
    }


def process_panorama(
    image_path: str,
    output_dir: str,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    处理单张全景图：分割并处理前后两部分
    """
    if config is None:
        config = get_default_config()
    
    start_time = time.time()
    image_path_obj = Path(image_path)
    basename = image_path_obj.stem
    
    print(f"\n{'='*60}")
    print(f"处理全景图: {image_path_obj.name}")
    print(f"{'='*60}")
    
    # 读取原始图片
    print("步骤1: 读取全景图...")
    original = cv2.imread(str(image_path_obj))
    if original is None:
        return {'success': False, 'error': f'无法读取图片: {image_path}'}
    
    height, width = original.shape[:2]
    print(f"  原始尺寸: {width}x{height}")
    
    # 裁剪并分割
    print("\n步骤2: 裁剪并分割全景图...")
    front_half, back_half = crop_and_split_panorama(original, bottom_crop_ratio=0.3803)
    print(f"  前半部分: {front_half.shape[1]}x{front_half.shape[0]}")
    print(f"  后半部分: {back_half.shape[1]}x{back_half.shape[0]}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    front_output = output_path / f"{basename}_front"
    back_output = output_path / f"{basename}_back"
    front_output.mkdir(parents=True, exist_ok=True)
    back_output.mkdir(parents=True, exist_ok=True)
    
    # 处理前半部分
    print("\n步骤3: 处理前半部分 (0-180°)...")
    t1 = time.time()
    front_result = process_half_image(
        front_half,
        f"{basename}_front",
        str(front_output),
        config
    )
    front_time = time.time() - t1
    print(f"  ✅ 前半部分完成 ({front_time:.2f}秒)")
    
    # 处理后半部分
    print("\n步骤4: 处理后半部分 (180-360°)...")
    t2 = time.time()
    back_result = process_half_image(
        back_half,
        f"{basename}_back",
        str(back_output),
        config
    )
    back_time = time.time() - t2
    print(f"  ✅ 后半部分完成 ({back_time:.2f}秒)")
    
    total_time = time.time() - start_time
    print(f"\n✅ 全部完成！总耗时: {total_time:.2f}秒")
    
    return {
        'success': True,
        'output_dir': output_dir,
        'front_output': str(front_output),
        'back_output': str(back_output),
        'total_time': total_time,
        'front_time': front_time,
        'back_time': back_time
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python main.py <图片路径> <输出目录>")
        print("示例: python main.py input/panorama.jpg output")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    result = process_panorama(image_path, output_dir)
    
    if not result['success']:
        print(f"\n❌ 处理失败: {result.get('error', 'Unknown error')}")
        sys.exit(1)


