# 桌子排列策略功能说明

## 概述
已将 `visualize_desk_detection.py` 中的桌子排列策略成功集成到 `standalone_detect.py` 中。

## 主要功能

### 1. 核心算法
- **分列算法**: 基于原点距离和方向约束的桌子分列策略 (`split_into_columns_by_origin_walk`)
- **直线拟合**: 对每列桌子进行直线拟合 (`fit_line_kb_positive`)
- **点线距离**: 计算点到拟合线的距离 (`point_line_distance_kb`)
- **线条裁剪**: 将拟合线裁剪到图像边界内 (`clip_line_kb_to_image`)

### 2. 分列策略
采用"原点距离+方向约束"的行走策略:
1. 第一列头点: x 最小且距离原点最近
2. 同列向上找点: x 变小 + y 变小 + 距离原点近
3. 下一列头点: 相对上一列头点 x 变大 + y 变大 且距离原点最近
4. 重复得到指定列数(默认5列)，每列目标6个点

### 3. 可视化增强
- 每个桌子显示座位编号 (第1列: 1-6, 第2列: 7-12, ...)
- 不同颜色标识不同列
- 绘制列内拟合直线
- 相邻桌子间绘制连线(左上角->左上角, 右下角->右下角)
- 显示每列的统计信息

## 使用方法

### 基本检测(不启用排列策略)
```bash
python exam_seat_binding/standalone_detect.py \
  --source data/desk/originimages/room_first_frame.png \
  --weights exam_seat_binding/weight/yolo11desk.pt
```

### 启用桌子排列策略
```bash
python exam_seat_binding/standalone_detect.py \
  --source data/desk/originimages/room_first_frame.png \
  --weights exam_seat_binding/weight/yolo11desk.pt \
  --enable-desk-layout
```

### 指定列数
```bash
python exam_seat_binding/standalone_detect.py \
  --source data/desk/originimages \
  --weights exam_seat_binding/weight/yolo11desk.pt \
  --enable-desk-layout \
  --num-cols 5
```

### 完整示例(包含所有参数)
```bash
python exam_seat_binding/standalone_detect.py \
  --source data/desk/originimages \
  --weights exam_seat_binding/weight/yolo11desk.pt \
  --enable-desk-layout \
  --num-cols 5 \
  --conf 0.6 \
  --iou 0.45 \
  --device 0 \
  --img-size 1280 \
  --output exam_seat_binding/outputs
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable-desk-layout` | 启用桌子排列策略(自动分列、编号和连线) | False |
| `--num-cols` | 桌子列数 | 5 |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS的IOU阈值 | 0.45 |
| `--device` | 运行设备(cpu/cuda/0/1等) | 自动选择 |
| `--img-size` | 推理图像尺寸 | None |
| `--output` | 结果保存目录 | output |

## 输出结果

### 不启用排列策略
- 输出文件名: `detected_<原文件名>`
- 显示内容: 基本检测框和置信度

### 启用排列策略
- 输出文件名: `detected_layout_<原文件名>`
- 显示内容:
  - 灰色底框(所有检测)
  - 彩色框(按列分组)
  - 座位编号(1-30)
  - 列内拟合直线
  - 相邻桌子连线
  - 列统计信息(控制台输出)

## 测试结果示例

```
第1列: 6个桌子
第2列: 6个桌子
第3列: 6个桌子
第4列: 6个桌子
第5列: 7个桌子
检测到 31 个目标 - 保存至: exam_seat_binding/outputs/detected_layout_room_first_frame.png
```

## 注意事项

1. 桌子排列策略仅适用于**图片检测**，对文件夹批量处理有效
2. 视频检测暂不支持排列策略(会自动回退到基本检测)
3. 当检测目标数量较少时，分列结果可能不均匀
4. 不同场景可能需要调整 `--num-cols` 参数
5. 启用排列策略后，输出文件名会添加 `_layout` 后缀以区分

## 技术细节

### 核心类修改
- `StandaloneDetector.__init__`: 添加 `enable_desk_layout` 和 `num_cols` 参数
- `StandaloneDetector.detect_image`: 集成分列算法调用
- 新增方法 `_draw_boxes_with_layout`: 绘制带排列策略的可视化结果

### 导入的函数
从 `visualize_desk_detection.py` 移植的核心函数:
- `fit_line_kb_positive`: 直线拟合
- `point_line_distance_kb`: 点线距离计算
- `clip_line_kb_to_image`: 线条裁剪
- `split_into_columns_by_origin_walk`: 分列算法
- 以及相关的辅助函数

## 未来改进方向

1. 添加误检测修复机制(`repair_columns_by_drop_refit`, `repair_columns_by_translation`)
2. 支持视频的排列策略检测
3. 添加更多分列算法选项(KMeans, RANSAC等)
4. 支持自定义座位编号规则
5. 输出JSON格式的检测结果和分列信息
