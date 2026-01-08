# 🔬 视觉分析Pipeline - 7阶段工作流程详细规范

**用途：** 指导代码编写的完整技术规范文档  
**目标：** 将1张输入图片处理成20张输出图片

---

## 📋 总览

```
输入: 1张图片 (image.jpg)
输出: 20张图片 + 元数据
```

### 阶段依赖关系图

```
阶段1 (预处理)
    ↓
阶段2 (AI推理)
    ├─ semantic_map ──→ 阶段3 (后处理) ──→ 阶段5 (开放度)
    └─ depth_map ────→ 阶段4 (景深分层)
                           ↓
                      阶段6 (生成20张图)
                           ↓
                      阶段7 (保存输出)
```

---

## 阶段1: 图片预处理

### 📥 输入

```
文件路径: string (例如: "input/photo.jpg")
```

### 📤 输出

```python
{
    'original': np.ndarray,        # 原始图片 (H, W, 3) BGR格式
    'original_copy': np.ndarray,   # 原始图片副本 (H, W, 3) BGR格式
    'height': int,                 # 图片高度
    'width': int,                  # 图片宽度
    'metadata': dict               # 元数据（文件名、尺寸等）
}
```

### 🔧 处理步骤

#### 1.1 读取图片文件

```
操作: 使用 OpenCV 或 PIL 读取图片
要求:
  - 统一为 BGR 颜色空间（OpenCV默认）
  - 保持原始尺寸，不进行缩放
  - 检查文件是否存在
```

**伪代码：**

```
function load_image(path):
    if not file_exists(path):
        raise FileNotFoundError
    
    image = cv2.imread(path)
    
    if image is None:
        raise ValueError("无法读取图片")
    
    return image
```

#### 1.2 创建副本

```
操作: 深拷贝原始图片
原因: 
  - original 用于阶段2的AI推理（可能被修改）
  - original_copy 用于阶段6的原图输出（保持不变）
```

**伪代码：**

```
original_copy = original.copy()  # 深拷贝
```

#### 1.3 提取图片属性

```
提取:
  - 高度 (height)
  - 宽度 (width)
  - 通道数（应该是3）
  - 数据类型（应该是 uint8）
```

**伪代码：**

```
height, width, channels = original.shape

assert channels == 3, "必须是3通道RGB/BGR图片"
assert original.dtype == np.uint8, "必须是8位图片"
```

#### 1.4 生成元数据

```
收集:
  - 文件名（不含路径）
  - 文件扩展名
  - 图片尺寸字符串（如 "1920x1080"）
  - 像素总数
  - 时间戳
```

**数据结构：**

```python
metadata = {
    'filename': 'photo.jpg',
    'basename': 'photo',
    'extension': '.jpg',
    'size_str': '1920x1080',
    'total_pixels': 2073600,
    'timestamp': '2025-01-01 12:00:00'
}
```

### ✅ 质量检查点

- [ ] 图片成功加载，无损坏
- [ ] original 和 original_copy 是两个独立对象
- [ ] height > 0 且 width > 0
- [ ] channels == 3
- [ ] dtype == uint8
- [ ] 元数据完整

### 🧪 测试建议

```
测试用例1: 正常图片
输入: 有效的 JPG/PNG 文件
预期: 成功返回数据

测试用例2: 无效文件
输入: 不存在的路径
预期: 抛出 FileNotFoundError

测试用例3: 损坏文件
输入: 损坏的图片文件
预期: 抛出 ValueError

测试用例4: 灰度图
输入: 单通道灰度图
预期: 转换为3通道或抛出错误
```

---

## 阶段2: AI模型推理

### 📥 输入

```python
{
    'image': np.ndarray,      # 来自阶段1的 original (H, W, 3)
    'config': {
        'classes': List[str],           # 语义类别列表
        'encoder': str,                 # 模型大小 ('vitb' 或 'vits')
        'class_colors': Dict[int, List[int]]  # 类别ID到BGR颜色的映射
    }
}
```

### 📤 输出

```python
{
    'semantic_map': np.ndarray,   # 语义分割图 (H, W) dtype=uint8
    'depth_map': np.ndarray       # 深度图 (H, W) dtype=uint8
}
```

### 🔧 处理步骤

#### 2.1 语义分割推理

##### 输入规格

```
image: (H, W, 3) BGR uint8
classes: ['sky', 'grass', 'tree', 'building', ...]
```

##### 输出规格

```
semantic_map: (H, W) uint8
值范围: [0, N]
  - 0: 背景/未分类
  - 1-N: 语义类别ID（N = len(classes)）
```

##### 处理流程

**步骤1: 初始化分割图**

```
semantic_map = np.zeros((H, W), dtype=np.uint8)
```

**步骤2: 对每个类别进行分割**

```
for class_id, class_name in enumerate(classes, start=1):
    # 使用 SAM 2.1 + LangSAM 进行文本引导分割
    
    步骤2.1: 准备文本提示
        text_prompt = class_name
    
    步骤2.2: 模型推理
        masks = model.predict(image, text=text_prompt)
        # masks: (N, H, W) bool - 可能有多个实例
    
    步骤2.3: 合并多个实例
        if len(masks) > 0:
            combined_mask = masks.any(axis=0)  # (H, W) bool
        else:
            continue  # 该类别未检测到
    
    步骤2.4: 写入分割图
        semantic_map[combined_mask] = class_id
```

**步骤3: 处理重叠**

```
注意: 后处理的类别会覆盖先处理的类别
建议: 按优先级排序类别（重要的类别放后面）
例如: ['ground', 'building', 'tree', 'person']
     （person最重要，最后处理，不会被覆盖）
```

##### 关键参数

```python
model_config = {
    'encoder': 'vitb',  # 或 'vits'
    'image_size': 1024,
    'conf_threshold': 0.3,  # 置信度阈值
    'box_threshold': 0.25
}
```

##### 边界条件处理

```
情况1: 某个类别未检测到
处理: 跳过，继续下一个类别

情况2: 多个类别重叠
处理: 后处理的覆盖先处理的

情况3: 所有类别都未检测到
处理: semantic_map 全为0（背景）
```

#### 2.2 深度估计推理

##### 输入规格

```
image: (H, W, 3) BGR uint8
```

##### 输出规格

```
depth_map: (H, W) uint8
值范围: [0, 255]
  - 0: 最近（前景）
  - 255: 最远（背景）
```

##### 处理流程

**步骤1: 模型推理**

```
使用 Depth Anything V2 进行深度估计

depth_raw = model.infer(image)
# depth_raw: (H, W) float32
# 值范围: 任意正浮点数
```

**步骤2: 归一化到 [0, 255]**

```
depth_min = depth_raw.min()
depth_max = depth_raw.max()

if depth_max == depth_min:
    # 边界情况: 图片深度完全一致（罕见）
    depth_normalized = np.full((H, W), 128, dtype=np.uint8)
else:
    depth_normalized = ((depth_raw - depth_min) / (depth_max - depth_min) * 255)
    depth_normalized = depth_normalized.astype(np.uint8)
```

**步骤3: 验证输出**

```
assert depth_normalized.min() == 0 或接近0
assert depth_normalized.max() == 255 或接近255
assert depth_normalized.dtype == np.uint8
```

##### 深度值含义

```
深度值越小 → 距离相机越近 → 前景
深度值越大 → 距离相机越远 → 背景

```

### ✅ 质量检查点

**语义分割检查:**

- [ ] semantic_map.shape == (H, W)
- [ ] semantic_map.dtype == np.uint8
- [ ] 0 <= semantic_map.max() <= len(classes)
- [ ] 至少有一个像素被分类（不是全0）

**深度估计检查:**

- [ ] depth_map.shape == (H, W)
- [ ] depth_map.dtype == np.uint8
- [ ] depth_map.min() >= 0
- [ ] depth_map.max() <= 255
- [ ] 深度值分布合理（不是全黑或全白）

### 🧪 测试建议

```
测试用例1: 简单场景（天空+草地）
输入: 上半部分蓝色，下半部分绿色
预期: 
  - semantic_map: 上半部分=天空ID，下半部分=草地ID
  - depth_map: 上半部分>下半部分（天空更远）

测试用例2: 复杂场景（多类别）
输入: 包含建筑、树木、人物的图片
预期: 
  - semantic_map: 所有类别都有一定数量的像素
  - depth_map: 近景物体值小，远景物体值大

测试用例3: 边界情况（单一颜色）
输入: 纯色图片
预期: 
  - semantic_map: 可能全为背景或单一类别
  - depth_map: 归一化后应有合理分布
```

---

## 阶段3: 后处理优化

### 📥 输入

```python
{
    'semantic_map': np.ndarray,    # 来自阶段2 (H, W) uint8
    'config': {
        'enable_hole_filling': bool,
        'enable_median_blur': bool,
        'hole_fill_kernel_size': int,  # 默认 5
        'blur_kernel_size': int        # 默认 5
    }
}
```

### 📤 输出

```python
{
    'semantic_map_processed': np.ndarray,  # 处理后的语义图 (H, W) uint8
    'processing_stats': {
        'holes_filled': int,
        'pixels_modified': int
    }
}
```

### 🔧 处理步骤

#### 3.1 智能空洞填充

##### 目的

```
填补语义分割图中的小空洞（值为0的未分类像素）
保持大结构不变
```

##### 算法: 形态学闭运算

**理论基础:**

```
闭运算 = 膨胀 + 腐蚀
效果: 
  - 填充小空洞（内部的0变成1）
  - 平滑凸起边界
  - 保持整体形状和大小
```

##### 处理流程

**步骤1: 记录处理前状态**

```
holes_before = np.sum(semantic_map == 0)
```

**步骤2: 创建形态学核**

```
kernel_size = 5  # 可配置
kernel_shape = 'ELLIPSE'  # 或 'RECT', 'CROSS'

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (kernel_size, kernel_size)
)
```

**步骤3: 对每个类别分别处理**

```
filled_map = semantic_map.copy()
num_classes = int(semantic_map.max())

for class_id in range(1, num_classes + 1):
    步骤3.1: 提取该类别的二值掩码
        class_mask = (semantic_map == class_id).astype(np.uint8)
        # class_mask: (H, W) uint8, 值为0或1
    
    步骤3.2: 闭运算
        closed_mask = cv2.morphologyEx(
            class_mask,
            cv2.MORPH_CLOSE,
            kernel
        )
        # closed_mask: (H, W) uint8, 值为0或1
    
    步骤3.3: 只更新原来是空洞的像素
        # 找到被闭运算填充的新像素
        new_pixels = (closed_mask == 1) & (filled_map == 0)
        
        # 将这些像素标记为该类别
        filled_map[new_pixels] = class_id
```

**步骤4: 统计**

```
holes_after = np.sum(filled_map == 0)
holes_filled = holes_before - holes_after
```

##### 参数调优

```
kernel_size:
  - 3: 轻度填充（只填最小空洞）
  - 5: 中度填充（推荐）
  - 7: 重度填充（可能改变形状）
  - 9+: 很强的填充（慎用）

kernel_shape:
  - ELLIPSE: 圆形，自然平滑（推荐）
  - RECT: 矩形，保持直角
  - CROSS: 十字形，方向性填充
```

##### 边界条件

```
情况1: semantic_map 没有空洞（全部被分类）
处理: 跳过填充，直接返回

情况2: semantic_map 全是空洞（全为0）
处理: 无法填充，保持不变，输出警告

情况3: 某个类别只有零散像素
处理: 闭运算可能连接这些像素
```

#### 3.2 中值滤波平滑

##### 目的

```
去除孤立噪点
平滑类别边界
减少"椒盐噪声"
```

##### 算法: 中值滤波

**理论基础:**

```
对每个像素，取其邻域窗口内所有值的中位数
效果:
  - 保留边缘
  - 去除孤立异常值
  - 平滑噪声
```

##### 处理流程

**步骤1: 确保kernel_size是奇数**

```
if kernel_size % 2 == 0:
    kernel_size += 1
    # 中值滤波要求奇数核（有中心点）
```

**步骤2: 应用中值滤波**

```
smoothed_map = cv2.medianBlur(
    semantic_map,
    ksize=kernel_size
)
```

**步骤3: 统计修改的像素**

```
pixels_modified = np.sum(smoothed_map != semantic_map)
```

##### 工作原理示例

```
5×5窗口示例:

原始值:            排序后:           结果:
1 1 1 1 1         [1,1,1,1,1,      中位数 = 1
1 1 1 1 1          1,1,1,1,1,      (第13个元素)
1 1 2 1 1    →     1,1,1,1,1,  →   
1 1 1 1 1          1,1,1,1,1,      中心点: 2 → 1
1 1 1 1 1          1,1,1,1,1,      (噪点被去除)
                   1,1,1,2]
```

##### 参数调优

```
kernel_size:
  - 3: 轻度平滑（保留细节）
  - 5: 中度平滑（推荐）
  - 7: 重度平滑（可能丢失细节）
  - 9+: 很强的平滑（边界模糊）
```

### 🔀 处理顺序

```
推荐顺序:
  1. 先填充空洞（hole_filling）
  2. 后中值滤波（median_blur）

原因:
  - 填充后的结果更连续，适合平滑
  - 平滑操作不会引入新空洞

可选顺序:
  - 只用填充
  - 只用平滑
  - 两个都不用（保持原始结果）
```

### ✅ 质量检查点

- [ ] processed_map.shape == semantic_map.shape
- [ ] processed_map.dtype == np.uint8
- [ ] processed_map 的类别ID范围没有变化
- [ ] 空洞数量减少（如果启用填充）
- [ ] 噪点减少（如果启用平滑）

### 🧪 测试建议

```
测试用例1: 有明显空洞的图
输入: semantic_map 中间有连续的0像素区域
预期: 
  - 空洞被周围类别填充
  - holes_filled > 0

测试用例2: 有孤立噪点的图
输入: semantic_map 中有零散的错误分类
预期:
  - 孤立点被周围主导类别替换
  - pixels_modified > 0

测试用例3: 完美的分割图
输入: semantic_map 无空洞无噪点
预期:
  - 输出与输入几乎相同
  - holes_filled ≈ 0, pixels_modified ≈ 0
```

---

## 阶段4: 景深分层

### 📥 输入

```python
{
    'depth_map': np.ndarray,  # 来自阶段2 (H, W) uint8, 值范围 [0, 255]
    'config': {
        'split_method': str,  # 'percentile' 或 'fixed_threshold'
        'fg_ratio': float,    # 前景比例 (默认 0.33)
        'bg_ratio': float     # 背景比例 (默认 0.33)
    }
}
```

### 📤 输出

```python
{
    'foreground_mask': np.ndarray,     # 前景掩码 (H, W) bool
    'middleground_mask': np.ndarray,   # 中景掩码 (H, W) bool
    'background_mask': np.ndarray,     # 背景掩码 (H, W) bool
    'depth_thresholds': {
        'P33': float,  # 前景/中景分界点
        'P66': float   # 中景/背景分界点
    },
    'layer_stats': {
        'foreground_pixels': int,
        'middleground_pixels': int,
        'background_pixels': int,
        'foreground_percent': float,
        'middleground_percent': float,
        'background_percent': float
    }
}
```

### 🔧 处理步骤

#### 4.1 计算深度分位数

##### 方法1: 百分位数划分（推荐）

**步骤1: 计算三分位点**

```
P33 = np.percentile(depth_map, 33)
P66 = np.percentile(depth_map, 66)

解释:
  - P33: 33%的像素深度 <= P33（前景/中景分界）
  - P66: 66%的像素深度 <= P66（中景/背景分界）
```

**步骤2: 创建掩码**

```
foreground_mask = (depth_map <= P33)
middleground_mask = (depth_map > P33) & (depth_map <= P66)
background_mask = (depth_map > P66)

类型: bool
形状: (H, W)
```

##### 方法2: 固定阈值划分（备选）

```
threshold_1 = 85   # 前景阈值 (0-85)
threshold_2 = 170  # 背景阈值 (171-255)

foreground_mask = (depth_map <= threshold_1)
middleground_mask = (depth_map > threshold_1) & (depth_map <= threshold_2)
background_mask = (depth_map > threshold_2)
```

#### 4.2 验证分层结果

**检查1: 完整性**

```
所有像素必须属于且仅属于一个层:

assert (foreground_mask | middleground_mask | background_mask).all()
# 每个像素至少属于一层

assert not (foreground_mask & middleground_mask).any()
assert not (middleground_mask & background_mask).any()
assert not (foreground_mask & background_mask).any()
# 没有像素属于多层
```

**检查2: 比例**

```
total_pixels = H * W

fg_pixels = foreground_mask.sum()
mg_pixels = middleground_mask.sum()
bg_pixels = background_mask.sum()

assert fg_pixels + mg_pixels + bg_pixels == total_pixels

fg_percent = fg_pixels / total_pixels * 100
mg_percent = mg_pixels / total_pixels * 100
bg_percent = bg_pixels / total_pixels * 100

# 理想情况: 33% / 34% / 33%
# 实际: 可能有偏差，取决于深度分布
```

#### 4.3 统计分析

```python
stats = {
    'foreground': {
        'pixels': int(fg_pixels),
        'percent': float(fg_percent),
        'depth_range': (depth_map[foreground_mask].min(), 
                       depth_map[foreground_mask].max()),
        'depth_mean': float(depth_map[foreground_mask].mean())
    },
    'middleground': {
        'pixels': int(mg_pixels),
        'percent': float(mg_percent),
        'depth_range': (depth_map[middleground_mask].min(), 
                       depth_map[middleground_mask].max()),
        'depth_mean': float(depth_map[middleground_mask].mean())
    },
    'background': {
        'pixels': int(bg_pixels),
        'percent': float(bg_percent),
        'depth_range': (depth_map[background_mask].min(), 
                       depth_map[background_mask].max()),
        'depth_mean': float(depth_map[background_mask].mean())
    }
}
```

### 🎨 可视化建议

```python
# 创建彩色的前中背景图（用于阶段6）
fmb_visualization = np.zeros((H, W, 3), dtype=np.uint8)

fmb_visualization[foreground_mask] = [0, 255, 0]    # 绿色 BGR
fmb_visualization[middleground_mask] = [0, 255, 255]  # 黄色 BGR
fmb_visualization[background_mask] = [255, 0, 0]    # 蓝色 BGR
```

### ✅ 质量检查点

- [ ] 三个掩码的形状都是 (H, W)
- [ ] 三个掩码的类型都是 bool
- [ ] 所有像素都被分配到某一层
- [ ] 没有像素属于多层
- [ ] 前景平均深度 < 中景平均深度 < 背景平均深度
- [ ] 每层至少有一些像素（不能为空）

### 🧪 测试建议

```
测试用例1: 均匀深度分布
输入: depth_map 值均匀分布在 [0, 255]
预期:
  - fg_percent ≈ 33%
  - mg_percent ≈ 34%
  - bg_percent ≈ 33%

测试用例2: 极端深度分布（全前景）
输入: depth_map 全部是小值（0-50）
预期:
  - fg_percent 可能接近 100%
  - mg_percent 和 bg_percent 很小

测试用例3: 双峰分布
输入: depth_map 一半是0，一半是255
预期:
  - fg 和 bg 各占约50%
  - mg 很小
```

---

## 阶段5: 开放度计算

### 📥 输入

```python
{
    'semantic_map': np.ndarray,  # 来自阶段3处理后 (H, W) uint8
    'config': {
        'classes': List[str],           # 类别列表
        'openness_config': List[int]    # 每个类别的开放度 [0或1]
    }
}
```

**配置示例:**

```python
classes = ['sky', 'grass', 'trees', 'building', 'water', 'person']
openness_config = [1,     1,       0,       0,          1,        0]
#                  开放   开放     封闭     封闭        开放      封闭
```

### 📤 输出

```python
{
    'openness_map': np.ndarray,  # 开放度图 (H, W) uint8
    'openness_stats': {
        'open_pixels': int,
        'closed_pixels': int,
        'openness_ratio': float  # 开放像素比例
    }
}
```

### 🔧 处理步骤

#### 5.1 验证配置

**检查1: 列表长度一致**

```
assert len(openness_config) == len(classes)

错误处理:
if len(openness_config) != len(classes):
    raise ValueError(
        f"配置不匹配: {len(classes)} 个类别, "
        f"但有 {len(openness_config)} 个开放度值"
    )
```

**检查2: 值有效性**

```
for i, value in enumerate(openness_config):
    assert value in [0, 1], f"类别 {classes[i]} 的开放度必须是0或1"
```

#### 5.2 计算开放度图

**步骤1: 初始化**

```
H, W = semantic_map.shape
openness_map = np.zeros((H, W), dtype=np.uint8)
```

**步骤2: 查表映射**

```
for class_id in range(1, len(classes) + 1):
    步骤2.1: 找到该类别的所有像素
        class_pixels = (semantic_map == class_id)
    
    步骤2.2: 查询该类别的开放度
        is_open = openness_config[class_id - 1]
        # 注意索引: class_id从1开始，但列表索引从0开始
    
    步骤2.3: 设置开放度值
        if is_open == 1:
            openness_map[class_pixels] = 255  # 开放=白色
        else:
            openness_map[class_pixels] = 0    # 封闭=黑色
```

**步骤3: 处理未分类像素（背景）**

```
unclassified = (semantic_map == 0)
if unclassified.any():
    # 选项1: 标记为封闭
    openness_map[unclassified] = 0
    
    # 选项2: 标记为开放（如果背景是天空等）
    # openness_map[unclassified] = 255
    
    # 选项3: 忽略（保持为0）
    pass
```

#### 5.3 统计开放度

**计算像素数量:**

```
open_pixels = np.sum(openness_map == 255)
closed_pixels = np.sum(openness_map == 0)
total_pixels = H * W

assert open_pixels + closed_pixels == total_pixels
```

**计算开放度比例:**

```
openness_ratio = open_pixels / total_pixels
# 值范围: [0.0, 1.0]
# 0.0 = 完全封闭
# 1.0 = 完全开放
```

**详细统计:**

```python
openness_stats = {
    'open_pixels': open_pixels,
    'closed_pixels': closed_pixels,
    'total_pixels': total_pixels,
    'openness_ratio': openness_ratio,
    'openness_percent': openness_ratio * 100,
    'by_class': {}
}

# 每个类别的开放度贡献
for class_id in range(1, len(classes) + 1):
    class_pixels = (semantic_map == class_id)
    class_count = class_pixels.sum()
    
    if class_count > 0:
        openness_stats['by_class'][classes[class_id-1]] = {
            'pixels': int(class_count),
            'is_open': bool(openness_config[class_id-1]),
            'contribution_to_openness': float(class_count / total_pixels)
        }
```

### 📊 开放度分级（可选）

```python
# 将连续的开放度转换为等级
def classify_openness(ratio):
    if ratio >= 0.7:
        return "高度开放"
    elif ratio >= 0.4:
        return "中度开放"
    elif ratio >= 0.1:
        return "低度开放"
    else:
        return "封闭"

openness_level = classify_openness(openness_ratio)
```

### ✅ 质量检查点

- [ ] openness_map.shape == semantic_map.shape
- [ ] openness_map.dtype == np.uint8
- [ ] openness_map 中只有0和255两个值
- [ ] open_pixels + closed_pixels == total_pixels
- [ ] 0.0 <= openness_ratio <= 1.0
- [ ] 配置与类别数量匹配

### 🧪 测试建议

```
测试用例1: 全开放场景
输入: 
  - semantic_map: 只有 sky 和 grass
  - openness_config: [1, 1] (都开放)
预期:
  - openness_map 全为255
  - openness_ratio = 1.0

测试用例2: 全封闭场景
输入:
  - semantic_map: 只有 building 和 tree
  - openness_config: [0, 0] (都封闭)
预期:
  - openness_map 全为0
  - openness_ratio = 0.0

测试用例3: 混合场景
输入:
  - semantic_map: 50% sky, 50% building
  - openness_config: [1, 0]
预期:
  - openness_ratio ≈ 0.5
```

---

## 阶段6: 生成20张图片

### 📥 输入

```python
{
    'original_copy': np.ndarray,      # 来自阶段1 (H, W, 3) BGR
    'semantic_map': np.ndarray,       # 来自阶段3 (H, W) uint8
    'depth_map': np.ndarray,          # 来自阶段2 (H, W) uint8
    'openness_map': np.ndarray,       # 来自阶段5 (H, W) uint8
    'foreground_mask': np.ndarray,    # 来自阶段4 (H, W) bool
    'middleground_mask': np.ndarray,  # 来自阶段4 (H, W) bool
    'background_mask': np.ndarray,    # 来自阶段4 (H, W) bool
    'config': {
        'colors': Dict[int, List[int]]  # 类别颜色映射
    }
}
```

### 📤 输出

```python
{
    # 20张图片的字典
    'images': Dict[str, np.ndarray]
    
    # 每张图片: (H, W, 3) BGR uint8
}
```

**输出图片清单:**

```
基础图 (4张):
  1. semantic_map
  2. depth_map
  3. openness_map
  4. fmb_map

掩码图 (3张):
  5. foreground_map
  6. middleground_map
  7. background_map

原图 (1张):
  8. original

组合分层图 (12张):
  9-11.  semantic × 景深 (foreground, middleground, background)
  12-14. depth × 景深
  15-17. openness × 景深
  18-20. original × 景深
```

### 🔧 处理步骤

#### 6.1 生成基础分析图（4张）

##### 图1: semantic_map (彩色语义分割图)

**输入:**

```
semantic_map: (H, W) uint8, 值 [0, N]
colors: {0: [0,0,0], 1: [B1,G1,R1], ...}
```

**输出:**

```
semantic_colored: (H, W, 3) BGR uint8
```

**处理:**

```python
def colorize_semantic(semantic_map, colors):
    H, W = semantic_map.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id, bgr_color in colors.items():
        mask = (semantic_map == class_id)
        colored[mask] = bgr_color
    
    return colored
```

##### 图2: depth_map (彩色深度图)

**输入:**

```
depth_map: (H, W) uint8, 值 [0, 255]
```

**输出:**

```
depth_colored: (H, W, 3) BGR uint8
```

**处理:**

```python
def colorize_depth(depth_map, colormap='INFERNO'):
    # 应用伪彩色映射
    if colormap == 'INFERNO':
        colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    elif colormap == 'JET':
        colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    elif colormap == 'VIRIDIS':
        colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_VIRIDIS)
    
    return colored
```

**颜色映射说明:**

```
INFERNO:
  0 (近)   → 深蓝/紫色
  128 (中) → 红色/橙色
  255 (远) → 黄色/白色

JET:
  0   → 蓝色
  128 → 绿色
  255 → 红色

VIRIDIS:
  0   → 深紫色
  128 → 绿色
  255 → 黄色
```

##### 图3: openness_map (开放度图)

**输入:**

```
openness_map: (H, W) uint8, 值 [0, 255]
```

**输出:**

```
openness_colored: (H, W, 3) BGR uint8
```

**处理:**

```python
def colorize_openness(openness_map):
    # 方法1: 转为BGR (灰度)
    colored = cv2.cvtColor(openness_map, cv2.COLOR_GRAY2BGR)
    
    # 方法2: 应用彩色映射（可选）
    # colored = cv2.applyColorMap(openness_map, cv2.COLORMAP_BONE)
    
    return colored
```

##### 图4: fmb_map (前中背景图)

**输入:**

```
foreground_mask: (H, W) bool
middleground_mask: (H, W) bool
background_mask: (H, W) bool
```

**输出:**

```
fmb_colored: (H, W, 3) BGR uint8
```

**处理:**

```python
def create_fmb_visualization(fg_mask, mg_mask, bg_mask):
    H, W = fg_mask.shape
    fmb = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 颜色方案1: 红绿蓝
    fmb[fg_mask] = [0, 255, 0]      # 绿色 = 前景
    fmb[mg_mask] = [0, 255, 255]    # 黄色 = 中景
    fmb[bg_mask] = [255, 0, 0]      # 蓝色 = 背景
    
    # 颜色方案2: 深浅渐变（可选）
    # fmb[fg_mask] = [100, 255, 100]   # 浅绿
    # fmb[mg_mask] = [200, 200, 100]   # 黄绿
    # fmb[bg_mask] = [255, 150, 100]   # 橙色
    
    return fmb
```

#### 6.2 生成掩码图（3张）

**输入:**

```
foreground_mask: (H, W) bool
middleground_mask: (H, W) bool
background_mask: (H, W) bool
```

**输出:**

```
foreground_map: (H, W) uint8, 值 [0, 255]
middleground_map: (H, W) uint8, 值 [0, 255]
background_map: (H, W) uint8, 值 [0, 255]
```

**处理:**

```python
def create_mask_images(fg_mask, mg_mask, bg_mask):
    # 布尔转整数: True → 255, False → 0
    fg_image = (fg_mask * 255).astype(np.uint8)
    mg_image = (mg_mask * 255).astype(np.uint8)
    bg_image = (bg_mask * 255).astype(np.uint8)
    
    return {
        'foreground_map': fg_image,
        'middleground_map': mg_image,
        'background_map': bg_image
    }
```

#### 6.3 原图（1张）

**输入:**

```
original_copy: (H, W, 3) BGR uint8
```

**输出:**

```
original: (H, W, 3) BGR uint8
```

**处理:**

```python
# 直接使用副本
images['original'] = original_copy.copy()
```

#### 6.4 生成组合分层图（12张）- **核心算法**

##### 概念

```
组合分层图 = 基础图 × 景深掩码

公式:
layered_image(x, y) = {
    base_image(x, y),  如果 mask(x, y) = True
    [0, 0, 0],         否则
}

效果: 只显示特定景深层的内容，其他部分变黑
```

##### 通用函数

```python
def apply_mask_to_image(image, mask):
    """
    将掩码应用到图片
    
    参数:
        image: (H, W, 3) BGR uint8
        mask: (H, W) bool
    
    返回:
        masked_image: (H, W, 3) BGR uint8
    """
    # 方法1: 直接索引
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    # 方法2: 广播乘法（等效）
    # mask_3d = mask[:, :, np.newaxis]  # (H, W, 1)
    # masked_image = image * mask_3d
    
    return masked_image
```

##### 生成12张分层图

```python
def generate_layered_images(semantic_colored, depth_colored, 
                            openness_colored, original,
                            fg_mask, mg_mask, bg_mask):
    """
    生成所有组合分层图
    
    返回:
        dict: 12张图片
    """
    results = {}
    
    # 定义4种基础图
    base_images = {
        'semantic': semantic_colored,
        'depth': depth_colored,
        'openness': openness_colored,
        'original': original
    }
    
    # 定义3种掩码
    masks = {
        'foreground': fg_mask,
        'middleground': mg_mask,
        'background': bg_mask
    }
    
    # 笛卡尔积: 4 × 3 = 12
    for base_name, base_image in base_images.items():
        for mask_name, mask in masks.items():
            # 应用掩码
            layered = apply_mask_to_image(base_image, mask)
            
            # 命名: base_mask 格式
            key = f"{base_name}_{mask_name}"
            results[key] = layered
    
    return results

# 生成的12张图:
# semantic_foreground, semantic_middleground, semantic_background
# depth_foreground, depth_middleground, depth_background
# openness_foreground, openness_middleground, openness_background
# original_foreground, original_middleground, original_background
```

##### 详细示例

**示例1: semantic_foreground**

```
输入:
  - semantic_colored: 彩色语义图
  - foreground_mask: 前景掩码

处理:
  result = np.zeros_like(semantic_colored)
  result[foreground_mask] = semantic_colored[foreground_mask]

结果:
  只有前景部分有颜色（语义分类），
  中景和背景都是黑色
```

**示例2: original_background**

```
输入:
  - original: 原图
  - background_mask: 背景掩码

处理:
  result = np.zeros_like(original)
  result[background_mask] = original[background_mask]

结果:
  只有背景部分显示原图内容，
  前景和中景都是黑色
```

#### 6.5 合并所有图片

```python
def generate_all_20_images(...):
    """主函数: 生成全部20张图片"""
    
    all_images = {}
    
    # 步骤1: 基础分析图 (4张)
    all_images['semantic_map'] = colorize_semantic(...)
    all_images['depth_map'] = colorize_depth(...)
    all_images['openness_map'] = colorize_openness(...)
    all_images['fmb_map'] = create_fmb_visualization(...)
    
    # 步骤2: 掩码图 (3张)
    mask_images = create_mask_images(...)
    all_images.update(mask_images)
    
    # 步骤3: 原图 (1张)
    all_images['original'] = original_copy
    
    # 步骤4: 组合分层图 (12张)
    layered = generate_layered_images(...)
    all_images.update(layered)
    
    # 验证
    assert len(all_images) == 20
    
    return all_images
```

### ✅ 质量检查点

**每张图片检查:**

- [ ] 形状: (H, W, 3)
- [ ] 类型: np.uint8
- [ ] 值范围: [0, 255]

**整体检查:**

- [ ] 总共20张图片
- [ ] 没有重复的key
- [ ] 所有掩码图只有0和255
- [ ] 分层图的非掩码区域全为黑色[0,0,0]

### 🧪 测试建议

```
测试用例1: 验证数量
预期: len(images) == 20

测试用例2: 验证分层逻辑
操作: 
  - 创建简单的semantic_colored (上半红，下半蓝)
  - 创建foreground_mask (只有上半部分)
  - 生成 semantic_foreground
预期:
  - 上半部分是红色
  - 下半部分是黑色

测试用例3: 验证掩码完整性
操作:
  - 叠加 semantic_foreground + semantic_middleground + semantic_background
预期:
  - 结果应该等于 semantic_map
```

---

## 阶段7: 保存输出

### 📥 输入

```python
{
    'images': Dict[str, np.ndarray],  # 来自阶段6的20张图片
    'output_dir': str,                # 输出目录路径
    'image_basename': str,            # 原图文件名（不含扩展名）
    'metadata': dict                  # 处理元数据
}
```

### 📤 输出

```
文件系统输出:
output_dir/
  ├── semantic_map.png
  ├── depth_map.png
  ├── openness_map.png
  ├── fmb_map.png
  ├── foreground_map.png
  ├── middleground_map.png
  ├── background_map.png
  ├── original.png
  ├── semantic_foreground.png
  ├── semantic_middleground.png
  ├── semantic_background.png
  ├── depth_foreground.png
  ├── depth_middleground.png
  ├── depth_background.png
  ├── openness_foreground.png
  ├── openness_middleground.png
  ├── openness_background.png
  ├── original_foreground.png
  ├── original_middleground.png
  ├── original_background.png
  └── metadata.json

返回值:
{
    'output_dir': str,
    'saved_files': List[str],  # 保存的文件路径列表
    'success': bool,
    'errors': List[str]        # 错误信息（如果有）
}
```

### 🔧 处理步骤

#### 7.1 准备输出目录

**步骤1: 创建目录**

```python
import os
from pathlib import Path

def prepare_output_directory(output_dir):
    """
    创建输出目录（如果不存在）
    """
    # 转为Path对象
    output_path = Path(output_dir)
    
    # 创建目录（包括父目录）
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 验证可写
    if not os.access(output_path, os.W_OK):
        raise PermissionError(f"无法写入目录: {output_path}")
    
    return output_path
```

**步骤2: 清理旧文件（可选）**

```python
def clean_directory(directory, pattern='*.png'):
    """
    删除目录中的旧图片（可选）
    """
    for file in directory.glob(pattern):
        file.unlink()
```

#### 7.2 保存图片文件

**步骤1: 确定图片顺序**

```python
# 定义保存顺序（按类别分组）
image_order = [
    # 基础分析图
    'semantic_map', 'depth_map', 'openness_map', 'fmb_map',
    
    # 掩码图
    'foreground_map', 'middleground_map', 'background_map',
    
    # 原图
    'original',
    
    # 语义分层
    'semantic_foreground', 'semantic_middleground', 'semantic_background',
    
    # 深度分层
    'depth_foreground', 'depth_middleground', 'depth_background',
    
    # 开放度分层
    'openness_foreground', 'openness_middleground', 'openness_background',
    
    # 原图分层
    'original_foreground', 'original_middleground', 'original_background'
]
```

**步骤2: 保存每张图片**

```python
def save_all_images(images, output_dir, image_basename='result'):
    """
    保存所有图片
    
    参数:
        images: Dict[str, np.ndarray]
        output_dir: str | Path
        image_basename: str (可选的文件名前缀)
    
    返回:
        saved_files: List[str]
    """
    output_path = Path(output_dir)
    saved_files = []
    errors = []
    
    for name in image_order:
        if name not in images:
            errors.append(f"缺失图片: {name}")
            continue
        
        try:
            # 构造文件路径
            # 选项1: 不带前缀
            filename = f"{name}.png"
            
            # 选项2: 带前缀
            # filename = f"{image_basename}_{name}.png"
            
            filepath = output_path / filename
            
            # 保存图片
            success = cv2.imwrite(str(filepath), images[name])
            
            if success:
                saved_files.append(str(filepath))
            else:
                errors.append(f"保存失败: {filepath}")
        
        except Exception as e:
            errors.append(f"保存 {name} 时出错: {str(e)}")
    
    return saved_files, errors
```

**步骤3: 验证保存结果**

```python
def verify_saved_files(saved_files):
    """
    验证所有文件都已保存且可读
    """
    for filepath in saved_files:
        # 检查文件存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件未保存: {filepath}")
        
        # 检查文件大小 > 0
        if os.path.getsize(filepath) == 0:
            raise ValueError(f"文件为空: {filepath}")
        
        # 可选: 尝试读取验证完整性
        test_image = cv2.imread(filepath)
        if test_image is None:
            raise ValueError(f"文件损坏: {filepath}")
```

#### 7.3 生成并保存元数据

**元数据结构:**

```python
metadata = {
    # 输入信息
    'input': {
        'filename': str,
        'size': str,  # "1920x1080"
        'total_pixels': int
    },
    
    # 处理配置
    'config': {
        'classes': List[str],
        'openness_config': List[int],
        'encoder': str,
        'enable_hole_filling': bool,
        'enable_median_blur': bool
    },
    
    # 处理统计
    'statistics': {
        'semantic': {
            'num_classes': int,
            'pixels_per_class': Dict[str, int]
        },
        'depth': {
            'min': int,
            'max': int,
            'mean': float
        },
        'layers': {
            'foreground_percent': float,
            'middleground_percent': float,
            'background_percent': float
        },
        'openness': {
            'openness_ratio': float,
            'open_pixels': int,
            'closed_pixels': int
        }
    },
    
    # 输出信息
    'output': {
        'output_dir': str,
        'total_images': int,
        'files': List[str]
    },
    
    # 时间戳
    'processing_time': {
        'start': str,  # ISO format
        'end': str,
        'duration_seconds': float
    },
    
    # 版本信息
    'version': {
        'pipeline_version': str,
        'opencv_version': str,
        'numpy_version': str
    }
}
```

**保存JSON:**

```python
import json
from datetime import datetime

def save_metadata(metadata, output_dir):
    """
    保存元数据为JSON文件
    """
    metadata_path = Path(output_dir) / 'metadata.json'
    
    # 添加保存时间
    metadata['saved_at'] = datetime.now().isoformat()
    
    # 保存
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return str(metadata_path)
```

#### 7.4 生成缩略图预览（可选）

```python
def create_thumbnail_grid(images, output_dir, grid_size=(5, 4)):
    """
    创建20张图片的缩略图网格预览
    
    参数:
        images: Dict[str, np.ndarray] - 20张图片
        output_dir: str
        grid_size: (cols, rows) - 默认5列4行
    
    输出:
        thumbnail_grid.png - 包含所有图片的网格预览
    """
    cols, rows = grid_size
    assert cols * rows >= 20
    
    # 获取单张图片尺寸
    sample_image = list(images.values())[0]
    H, W = sample_image.shape[:2]
    
    # 缩略图尺寸（缩小到原图的1/4）
    thumb_h, thumb_w = H // 4, W // 4
    
    # 创建网格画布
    grid_h = rows * thumb_h
    grid_w = cols * thumb_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 填充每个格子
    for i, (name, image) in enumerate(images.items()):
        if i >= cols * rows:
            break
        
        row = i // cols
        col = i % cols
        
        # 缩放图片
        thumb = cv2.resize(image, (thumb_w, thumb_h))
        
        # 添加文字标签
        cv2.putText(
            thumb,
            name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # 放置到网格
        y1 = row * thumb_h
        y2 = y1 + thumb_h
        x1 = col * thumb_w
        x2 = x1 + thumb_w
        grid[y1:y2, x1:x2] = thumb
    
    # 保存网格图
    grid_path = Path(output_dir) / 'thumbnail_grid.png'
    cv2.imwrite(str(grid_path), grid)
    
    return str(grid_path)
```

#### 7.5 生成处理报告（可选）

```python
def generate_report(metadata, output_dir):
    """
    生成文本格式的处理报告
    """
    report_path = Path(output_dir) / 'report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("视觉分析处理报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 输入信息
        f.write("输入图片:\n")
        f.write(f"  文件名: {metadata['input']['filename']}\n")
        f.write(f"  尺寸: {metadata['input']['size']}\n")
        f.write(f"  像素数: {metadata['input']['total_pixels']:,}\n\n")
        
        # 处理结果
        f.write("处理结果:\n")
        f.write(f"  语义类别数: {metadata['statistics']['semantic']['num_classes']}\n")
        f.write(f"  开放度: {metadata['statistics']['openness']['openness_ratio']:.1%}\n")
        f.write(f"  前景占比: {metadata['statistics']['layers']['foreground_percent']:.1%}\n")
        f.write(f"  中景占比: {metadata['statistics']['layers']['middleground_percent']:.1%}\n")
        f.write(f"  背景占比: {metadata['statistics']['layers']['background_percent']:.1%}\n\n")
        
        # 输出信息
        f.write("输出文件:\n")
        f.write(f"  总计: {metadata['output']['total_images']} 张图片\n")
        f.write(f"  位置: {metadata['output']['output_dir']}\n\n")
        
        # 处理时间
        f.write("处理时间:\n")
        f.write(f"  耗时: {metadata['processing_time']['duration_seconds']:.2f} 秒\n")
    
    return str(report_path)
```

### ✅ 质量检查点

**文件系统检查:**

- [ ] output_dir 存在且可写
- [ ] 20张PNG图片都已保存
- [ ] metadata.json 已保存
- [ ] 所有文件大小 > 0

**图片完整性检查:**

- [ ] 每张图片都可以被 cv2.imread 读取
- [ ] 图片尺寸正确
- [ ] 图片内容符合预期

**元数据检查:**

- [ ] JSON格式正确
- [ ] 所有必要字段都存在
- [ ] 统计数据准确

### 🧪 测试建议

```
测试用例1: 正常保存
操作: 保存20张图片到新目录
预期:
  - 目录创建成功
  - 20个PNG文件存在
  - metadata.json 存在

测试用例2: 目录已存在
操作: 保存到已有文件的目录
预期:
  - 旧文件被覆盖（或保留，取决于策略）
  - 新文件正常保存

测试用例3: 权限不足
操作: 保存到只读目录
预期:
  - 抛出 PermissionError
  - 不创建任何文件

测试用例4: 磁盘空间不足
操作: 保存到空间不足的磁盘
预期:
  - 保存失败
  - 返回错误信息
```

---

## 🎯 完整Pipeline集成

### 主控制流程

```python
def process_image_pipeline(image_path, output_dir, config):
    """
    完整的7阶段Pipeline
    
    参数:
        image_path: str - 输入图片路径
        output_dir: str - 输出目录
        config: dict - 配置参数
    
    返回:
        result: dict - 处理结果
    """
    import time
    start_time = time.time()
    
    try:
        # ========== 阶段1: 预处理 ==========
        stage1_result = stage1_preprocess(image_path)
        original = stage1_result['original']
        original_copy = stage1_result['original_copy']
        H, W = stage1_result['height'], stage1_result['width']
        
        # ========== 阶段2: AI推理 ==========
        stage2_result = stage2_ai_inference(original, config)
        semantic_map = stage2_result['semantic_map']
        depth_map = stage2_result['depth_map']
        
        # ========== 阶段3: 后处理 ==========
        stage3_result = stage3_postprocess(semantic_map, config)
        semantic_map_processed = stage3_result['semantic_map_processed']
        
        # ========== 阶段4: 景深分层 ==========
        stage4_result = stage4_depth_layering(depth_map, config)
        fg_mask = stage4_result['foreground_mask']
        mg_mask = stage4_result['middleground_mask']
        bg_mask = stage4_result['background_mask']
        
        # ========== 阶段5: 开放度计算 ==========
        stage5_result = stage5_openness(semantic_map_processed, config)
        openness_map = stage5_result['openness_map']
        
        # ========== 阶段6: 生成20张图片 ==========
        stage6_result = stage6_generate_images(
            original_copy,
            semantic_map_processed,
            depth_map,
            openness_map,
            fg_mask, mg_mask, bg_mask,
            config
        )
        all_images = stage6_result['images']
        
        # ========== 阶段7: 保存输出 ==========
        # 收集元数据
        metadata = {
            'input': stage1_result['metadata'],
            'config': config,
            'statistics': {
                'semantic': stage3_result.get('stats', {}),
                'layers': stage4_result['layer_stats'],
                'openness': stage5_result['openness_stats']
            },
            'processing_time': {
                'start': time.strftime('%Y-%m-%d %H:%M:%S', 
                                      time.localtime(start_time)),
                'end': time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': time.time() - start_time
            }
        }
        
        # 保存
        stage7_result = stage7_save_outputs(
            all_images,
            output_dir,
            Path(image_path).stem,
            metadata
        )
        
        # 返回完整结果
        return {
            'success': True,
            'output_dir': output_dir,
            'saved_files': stage7_result['saved_files'],
            'metadata': metadata,
            'images': all_images  # 可选: 返回图片数据
        }
        
    except Exception as e:
        # 错误处理
        return {
            'success': False,
            'error': str(e),
            'stage': '检测错误发生的阶段'
        }
```

### 使用示例

```python
# 配置
config = {
    'classes': ['sky', 'grass', 'tree', 'building'],
    'openness_config': [1, 1, 0, 0],
    'colors': {
        0: [0, 0, 0],
        1: [255, 200, 150],
        2: [100, 255, 100],
        3: [50, 150, 50],
        4: [120, 120, 180]
    },
    'encoder': 'vitb',
    'enable_hole_filling': True,
    'enable_median_blur': True
}

# 运行
result = process_image_pipeline(
    image_path='input/photo.jpg',
    output_dir='output/photo_results/',
    config=config
)

if result['success']:
    print(f"✅ 处理成功！")
    print(f"输出目录: {result['output_dir']}")
    print(f"生成文件: {len(result['saved_files'])} 个")
else:
    print(f"❌ 处理失败: {result['error']}")
```

---

## 📚 附录

### A. 数据类型规范

```python
# NumPy数组类型
np.ndarray:
  - dtype: uint8, float32, bool
  - shape: (H, W) 或 (H, W, 3)
  - 值范围: 根据具体用途

# 颜色格式
BGR: [B, G, R]  # OpenCV默认
RGB: [R, G, B]  # PIL/matplotlib
HSV: [H, S, V]  # 色调/饱和度/明度

# 掩码类型
bool: True/False
uint8: 0/255
```

### B. 错误处理策略

```python
# 每个阶段都应包含:
try:
    # 主要处理逻辑
    result = process(...)
    
    # 验证输出
    validate(result)
    
    return result

except FileNotFoundError as e:
    # 文件相关错误
    log.error(f"文件错误: {e}")
    raise

except ValueError as e:
    # 数据验证错误
    log.error(f"数据错误: {e}")
    raise

except Exception as e:
    # 未知错误
    log.error(f"未知错误: {e}")
    raise
```

### C. 性能优化建议

```
1. 使用NumPy向量化操作（避免Python循环）
2. 大图片考虑分块处理
3. 使用内存映射处理超大图片
4. 缓存重复计算的结果
5. 并行处理多张图片
```

### D. 代码质量检查清单

```
□ 所有函数都有类型提示
□ 所有函数都有文档字符串
□ 关键步骤有注释
□ 使用有意义的变量名
□ 避免魔法数字（使用常量）
□ 每个阶段有单元测试
□ 有集成测试
□ 有性能基准测试
```

---

**文档版本:** 1.0  
**最后更新:** 2025-01-01  
**适用范围:** 视觉分析Pipeline开发指南