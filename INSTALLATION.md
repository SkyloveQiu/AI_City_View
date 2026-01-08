# 安装指南 - 精度优先方案

本文档说明如何安装精度优先方案所需的依赖包。

## 方案选择

**已选择: 精度优先方案**
- **语义分割**: SAM 2.1 + LangSAM
- **深度估计**: Depth Anything V2

## 系统要求

- Python >= 3.8
- CUDA >= 11.0 (推荐 11.8+)
- GPU内存 >= 8GB (推荐 16GB+)
- PyTorch >= 2.0

## 安装步骤

### 1. 基础依赖

```bash
# 安装基础库
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install tqdm>=4.65.0
```

### 2. PyTorch

```bash
# 根据你的CUDA版本选择
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本 (不推荐，速度很慢)
pip install torch torchvision torchaudio
```

### 3. SAM 2.1 + LangSAM

**方法1: 使用lang-sam包 (推荐)**

```bash
# 安装lang-sam
pip install lang-sam

# 或者从源码安装
git clone https://github.com/luca-medeiros/lang-segment-anything.git
cd lang-segment-anything
pip install -e .
```

**方法2: 使用segment-anything-2**

```bash
# 安装segment-anything-2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 下载模型权重
# ViT-S: https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_small.pt
# ViT-B: https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_base_plus.pt
```

**注意**: 具体安装方法需要根据SAM 2.1的最新发布情况调整。请查看:
- https://github.com/facebookresearch/segment-anything-2
- https://github.com/luca-medeiros/lang-segment-anything

### 4. Depth Anything V2

```bash
# 方法1: 从源码安装
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt

# 下载模型权重
# https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
# https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf
```

**或者使用Hugging Face:**

```bash
pip install transformers
pip install accelerate
```

**注意**: Depth Anything V2可能需要额外的依赖，请查看官方仓库:
- https://github.com/DepthAnything/Depth-Anything-V2

## 验证安装

创建测试脚本 `test_installation.py`:

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")

# 测试SAM 2.1 + LangSAM
try:
    from lang_sam import LangSAM
    print("✅ LangSAM导入成功")
except ImportError as e:
    print(f"❌ LangSAM导入失败: {e}")

# 测试Depth Anything V2
try:
    # 根据实际包名调整
    # from depth_anything_v2 import DepthAnythingV2
    print("✅ Depth Anything V2导入成功")
except ImportError as e:
    print(f"❌ Depth Anything V2导入失败: {e}")
```

运行:
```bash
python test_installation.py
```

## 常见问题

### 1. CUDA版本不匹配

**问题**: PyTorch CUDA版本与系统CUDA版本不匹配

**解决**: 
```bash
# 检查系统CUDA版本
nvcc --version

# 安装匹配的PyTorch版本
# 参考: https://pytorch.org/get-started/locally/
```

### 2. GPU内存不足

**问题**: GPU内存不足，无法加载模型

**解决**:
- 使用ViT-S编码器而非ViT-B (内存占用减半)
- 减小输入图像尺寸
- 使用CPU推理 (非常慢，不推荐)

### 3. 模型权重下载失败

**问题**: 模型权重下载超时或失败

**解决**:
- 使用镜像站点
- 手动下载权重文件
- 使用VPN

## 性能优化建议

1. **使用ViT-S编码器**: 速度提升约2倍，内存占用减半，精度略降
2. **混合精度推理**: 使用FP16可以节省约50%内存，速度提升约30%
3. **批处理**: 如果有多个类别，可以尝试批量处理
4. **多GPU并行**: 使用多个GPU并行处理不同图片

## 下一步

安装完成后，需要:
1. 下载模型权重文件
2. 在 `pipeline/stage2_ai_inference.py` 中实现实际的模型调用代码
3. 运行测试验证功能

参考文档:
- `research_analysis.md`: 性能分析
- `plan.md`: 技术规范


