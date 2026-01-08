# 视觉分析Pipeline

将1张输入图片处理成20张输出图片的完整Pipeline系统。

## 项目结构

```
AI城市景观/
├── main.py                 # 主入口
├── plan.md                 # 详细技术规范文档
├── research_analysis.md     # 算法研究与性能分析
├── requirements.txt        # 依赖包
├── README.md              # 本文件
└── pipeline/              # Pipeline模块
    ├── __init__.py
    ├── stage1_preprocess.py      # 阶段1: 图片预处理
    ├── stage2_ai_inference.py    # 阶段2: AI模型推理 (待实现)
    ├── stage3_postprocess.py     # 阶段3: 后处理优化 (待实现)
    ├── stage4_depth_layering.py  # 阶段4: 景深分层 (待实现)
    ├── stage5_openness.py        # 阶段5: 开放度计算 (待实现)
    ├── stage6_generate_images.py # 阶段6: 生成20张图片 (待实现)
    └── stage7_save_outputs.py    # 阶段7: 保存输出 (待实现)
```

## 功能概述

### 7个处理阶段

1. **阶段1: 图片预处理** ✅
   - 读取图片文件
   - 创建副本
   - 提取属性
   - 生成元数据

2. **阶段2: AI模型推理** ⚠️ (代码框架已就绪，需安装模型)
   - 语义分割: SAM 2.1 + LangSAM ✅ (精度优先方案)
   - 深度估计: Depth Anything V2 ✅ (精度优先方案)

3. **阶段3: 后处理优化** ⏳
   - 智能空洞填充
   - 中值滤波平滑

4. **阶段4: 景深分层** ⏳
   - 前景/中景/背景分层

5. **阶段5: 开放度计算** ⏳
   - 基于语义类别的开放度映射

6. **阶段6: 生成20张图片** ⏳
   - 基础分析图 (4张)
   - 掩码图 (3张)
   - 原图 (1张)
   - 组合分层图 (12张)

7. **阶段7: 保存输出** ⏳
   - 保存20张PNG图片
   - 生成元数据JSON

## 安装

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装AI模型 (精度优先方案)
# 详细步骤见 INSTALLATION.md
# - SAM 2.1 + LangSAM
# - Depth Anything V2
```

**重要**: 安装AI模型需要GPU和CUDA支持，详细安装步骤请查看 `INSTALLATION.md`

## 使用方法

```bash
# 基本用法
python main.py <图片路径> [输出目录]

# 示例
python main.py input/photo.jpg output/
```

## 配置

默认会自动读取仓库根目录的 `Semantic_configuration.json` 作为语义类别/颜色/开放度配置。

如需指定其它路径，可设置环境变量:

```bash
export PIPELINE_SEMANTIC_CONFIG=/path/to/Semantic_configuration.json
```

也可以在 `main.py` 中手动修改配置:

```python
config = {
    'classes': ['sky', 'grass', 'tree', 'building'],
    'openness_config': [1, 1, 0, 0],
    'colors': {...},
    'encoder': 'vitb',
    'enable_hole_filling': True,
    'enable_median_blur': True,
    ...
}
```

## 性能考虑

- **项目规模**: 140万张图片
- **当前状态**: 阶段1已实现，其他阶段待实现
- **性能优化**: 见 `research_analysis.md`

## 开发状态

- [x] 阶段1: 预处理
- [ ] 阶段2: AI推理
- [ ] 阶段3: 后处理
- [ ] 阶段4: 景深分层
- [ ] 阶段5: 开放度计算
- [ ] 阶段6: 生成图片
- [ ] 阶段7: 保存输出

## 测试

```bash
# 测试阶段1
python pipeline/stage1_preprocess.py <图片路径>
```

## 文档

- `plan.md`: 完整的技术规范文档 (2109行)
- `research_analysis.md`: 算法研究与性能分析

## 许可证

待定

