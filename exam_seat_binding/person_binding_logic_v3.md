# 人桌绑定 V3 当前逻辑

## 1. 固定桌子模型

`PersonDeskBindingPipelineV3` 先调用 `DeskLayoutDetector.select_best_layout_from_video()` 建立桌子布局。

- `--desk-reference-seconds N` 大于 0 时，只使用视频开头 N 秒采样桌子检测结果。
- 建模完成后，后续整段人绑定都固定使用这份桌子模型，不再逐帧更新桌子。
- 人绑定从第 `N * fps` 帧开始，前 N 秒只用于桌子建模。
- `--reference-sample-step` 控制桌子建模采样步长。
- `--desk-mode auto/scheme1/scheme2/normal`、`--desk-conf`、`--desk-iou`、`--desk-num-cols`、`--desk-required-per-col` 继续沿用桌子检测器参数。

## 2. 座位区域构建

桌子布局得到 5 列 x 6 行后，`InterDeskZoneBuilder` 为每张桌子生成一个梯形绑定区域。

- 同列相邻两张桌子之间的区域作为座位绑定区。
- 第一排向近相机方向扩展，参数 `--first-extend`。
- 最后一排向远相机方向扩展，参数 `--last-extend`。
- 所有桌子外轮廓再生成 `classroom_polygon`，默认只连接最边缘桌子，不再向外扩展；只有脚点在该区域内的人才参与学生绑定。
- 如需让包围区外扩，可以手动设置 `--classroom-padding`。

## 3. 人检测与初筛

逐帧使用人物 YOLO + ByteTrack：

- 每个人框取脚点 `foot_point = ((x1+x2)/2, y2)` 作为位置锚点。
- 脚点在 `classroom_polygon` 内，才可能是学生。
- 视频后半段启用 `MovementTeacherDetector`，移动路径和净位移过大的目标会被判为老师。
- 未绑定且仍在老师观察窗口内的目标，会延迟进入绑定，避免把走动老师误绑定到座位。

## 4. 主绑定：梯形 IoU 贪心匹配

`ZoneBinder.assign_batch()` 对当前帧学生候选做一对一匹配。

- 把人框转成四边形。
- 计算人框与每个梯形座位区的凸多边形 IoU 和交集面积。
- 只保留交集面积大于 1 的候选。
- 候选按 `IoU 高 -> 交集面积大 -> 脚点离区域中心近` 排序。
- 贪心匹配，保证同一帧一个人只占一个座位，一个座位只给一个人。

## 5. 兜底绑定：同列顺序匹配

如果某个人没有被 IoU 匹配，且当前没有稳定座位，会进入 `ColumnFallbackAssigner`。

- 根据脚点到列中心线的距离选择最近列。
- 距离超过列阈值则跳过。
- 将脚点投影到列方向，得到深度顺序。
- 同一列内按深度顺序与剩余空座位做保序匹配。
- 深度差过大则跳过，不强行绑定。

## 6. 稳定绑定与防跳变

`StableSeatManager` 负责把帧级候选座位变成稳定座位。

- 新座位需要连续命中 `initial_bind_seconds` 才确认。
- 已绑定后，如果人框仍和原座位区有交集，会继续保持原座位。
- 换座需要连续命中 `switch_seconds`，默认内部为 4.5 秒。
- 短时丢检保留座位 `miss_hold_seconds`，默认内部为 8 秒。
- 同一时刻一个座位只允许一个活跃 track 占用。

## 7. 最终输出

运行结束后按全视频累计票数生成最终绑定。

- `seat_track_votes[seat_no]` 统计每个座位被哪些 raw track 占用过。
- 每个座位取票数最多的 raw track 作为最终学生。
- 对外统一学生 ID 使用座位号 1..30。
- JSON 输出包含固定桌子模型、梯形区域、最终座位绑定和配置参数。

## 主要可优化点

- 桌子建模：`--desk-reference-seconds`、`--reference-sample-step`、`--desk-conf`。
- 座位区域范围：`--first-extend`、`--last-extend`、`--classroom-padding`。
- 人检测召回：`--conf`、`--iou`、`--tracker`。
- 老师过滤：`MovementTeacherDetector` 的移动距离、净位移、观察窗口。
- 防跳变：`StableSeatManager` 的初次确认、换座确认、丢检保持时间。
- 兜底绑定：列距离阈值、深度差阈值、保序策略。
