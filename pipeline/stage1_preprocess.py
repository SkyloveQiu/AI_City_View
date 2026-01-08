"""
阶段1: 图片预处理
功能: 读取图片文件，创建副本，提取属性，生成元数据
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def stage1_preprocess(image_path: str) -> Dict[str, Any]:
    """
    阶段1: 图片预处理
    
    参数:
        image_path: str - 输入图片路径
    
    返回:
        dict - 包含以下键:
            - original: np.ndarray - 原始图片 (H, W, 3) BGR格式
            - original_copy: np.ndarray - 原始图片副本 (H, W, 3) BGR格式
            - height: int - 图片高度
            - width: int - 图片宽度
            - metadata: dict - 元数据
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 无法读取图片或图片格式不正确
    """
    # ========== 步骤1: 读取图片文件 ==========
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 使用OpenCV读取图片 (BGR格式)
    original = cv2.imread(str(image_path_obj))
    
    if original is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # ========== 步骤2: 创建副本 ==========
    # 深拷贝原始图片
    original_copy = original.copy()
    
    # ========== 步骤3: 提取图片属性 ==========
    height, width, channels = original.shape
    
    # 验证图片格式
    if channels != 3:
        raise ValueError(f"图片必须是3通道 (RGB/BGR), 当前通道数: {channels}")
    
    if original.dtype != np.uint8:
        raise ValueError(f"图片必须是8位格式, 当前类型: {original.dtype}")
    
    if height <= 0 or width <= 0:
        raise ValueError(f"图片尺寸无效: {width}x{height}")
    
    # ========== 步骤4: 生成元数据 ==========
    # 提取文件名信息
    filename = image_path_obj.name
    basename = image_path_obj.stem
    extension = image_path_obj.suffix
    
    # 计算像素总数
    total_pixels = height * width
    
    # 生成尺寸字符串
    size_str = f"{width}x{height}"
    
    # 时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    metadata = {
        'filename': filename,
        'basename': basename,
        'extension': extension,
        'size_str': size_str,
        'width': width,
        'height': height,
        'total_pixels': total_pixels,
        'channels': channels,
        'dtype': str(original.dtype),
        'timestamp': timestamp,
        'file_path': str(image_path_obj.absolute())
    }
    
    # ========== 返回结果 ==========
    result = {
        'original': original,
        'original_copy': original_copy,
        'height': height,
        'width': width,
        'metadata': metadata
    }
    
    # ========== 质量检查 ==========
    _validate_stage1_result(result)
    
    return result


def _validate_stage1_result(result: Dict[str, Any]) -> None:
    """
    验证阶段1的输出结果
    
    参数:
        result: dict - 阶段1的输出结果
    
    异常:
        AssertionError: 验证失败
    """
    original = result['original']
    original_copy = result['original_copy']
    height = result['height']
    width = result['width']
    metadata = result['metadata']
    
    # 检查图片成功加载
    assert original is not None, "original 不能为 None"
    assert original_copy is not None, "original_copy 不能为 None"
    
    # 检查 original 和 original_copy 是两个独立对象
    assert original is not original_copy, "original 和 original_copy 必须是独立对象"
    assert not np.shares_memory(original, original_copy), "original 和 original_copy 必须不共享内存"
    
    # 检查尺寸
    assert height > 0, f"height 必须 > 0, 当前值: {height}"
    assert width > 0, f"width 必须 > 0, 当前值: {width}"
    
    # 检查通道数
    assert original.shape[2] == 3, f"通道数必须是3, 当前值: {original.shape[2]}"
    assert original_copy.shape[2] == 3, f"original_copy 通道数必须是3"
    
    # 检查数据类型
    assert original.dtype == np.uint8, f"dtype 必须是 uint8, 当前值: {original.dtype}"
    assert original_copy.dtype == np.uint8, f"original_copy dtype 必须是 uint8"
    
    # 检查形状一致性
    assert original.shape == original_copy.shape, "original 和 original_copy 形状必须一致"
    
    # 检查元数据完整性
    required_keys = ['filename', 'basename', 'extension', 'size_str', 
                     'width', 'height', 'total_pixels', 'timestamp']
    for key in required_keys:
        assert key in metadata, f"元数据缺少必要字段: {key}"
    
    # 检查元数据值
    assert metadata['width'] == width, "元数据中的 width 与图片实际宽度不一致"
    assert metadata['height'] == height, "元数据中的 height 与图片实际高度不一致"
    assert metadata['total_pixels'] == height * width, "元数据中的 total_pixels 计算错误"


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        try:
            result = stage1_preprocess(test_image_path)
            print("✅ 阶段1测试成功!")
            print(f"图片尺寸: {result['width']}x{result['height']}")
            print(f"总像素数: {result['metadata']['total_pixels']:,}")
            print(f"元数据: {result['metadata']}")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法: python stage1_preprocess.py <图片路径>")
        print("示例: python stage1_preprocess.py test_image.jpg")


