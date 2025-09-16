## Waymo_NOTR 代码审阅报告（聚焦 Waymo PVPS 数据准备）

本报告面向需要基于 Waymo_NOTR 进行 Waymo Panoptic Video Segmentation（PVPS）数据准备的开发者，旨在：
- 梳理项目结构，定位与 PVPS 强相关的脚本与函数；
- 区分可复用的 PVPS 代码与历史 NeRF 相关遗留代码；
- 解释 PVPS 数据流与关键脚本的输入/输出；
- 提供面向重构的具体建议与可移除项清单。


## 1. 项目结构总览

```
Waymo_NOTR/
├─ data/                         # 数据清单与分割（split），以及数据目录占位
│  ├─ waymo_train_list.txt       # Waymo training set 全部 segment 名称（按字母序）
│  ├─ waymo_valid_list.txt       # Waymo validation set 全部 segment 名称（按字母序）
│  ├─ waymo_splits/              # 预定义的下载/处理分割列表（demo, static32, dynamic32, diverse56 等）
│  └─ waymo/                     # 原始与处理后数据的推荐目录结构（raw/ processed/）
│     ├─ raw/{training,validation}/         # 存放下载的 TFRecord
│     └─ processed/{training,validation}/   # 转换后的可直接使用的数据组织
│
├─ datasets/                     # 与数据准备直接相关的脚本目录（PVPS 重点）
│  ├─ download_waymo.py          # 并发下载 Waymo TFRecord（支持 split / scene_ids）
│  ├─ waymo_converter.py         # 将 TFRecord 解析为图像/标定/位姿/点云投影/动态掩码等
│  ├─ generate_lidar_depth.py    # 从保存的 pointcloud.npz 为单个序列生成稀疏 LiDAR depth
│  ├─ base/                      # 通用数据源/数据集抽象（更多偏向训练/旧项目）
│  └─ pvps/                      # 与 PVPS 标注列表、统计、批量深度生成相关的工具
│     ├─ WOD/                    # Waymo 官方 PVPS 帧列表与 camera_segmentation.proto
│     │  ├─ 2d_pvps_training_frames.txt
│     │  ├─ 2d_pvps_validation_frames.txt
│     │  └─ camera_segmentation.proto
│     ├─ pvps_count.py           # 统计各 split 中带标注帧数 -> 输出每 segment 的计数
│     ├─ find_segment_indices.py # 将带标注的 segment 名映射到 master 列表的行号（索引）
│     └─ generate_depth.py       # 批量遍历 processed/{split} 下的多个序列生成 LiDAR depth
│
├─ deeplab2/                     # DeepLab2 源码（可用于语义分割推理，如果不用 GT 语义）
│  └─ ...
│
├─ submodules/
│  └─ simple-waymo-open-dataset-reader/   # 第三方 Waymo 读取器（简化版，含 proto 生成脚本）
│
├─ utils/                        # 公用工具函数（PVPS 用到投影/框处理；大量 NeRF 遗留）
│  ├─ waymo_utils.py             # 大量 NeRF/Colmap/点云构建流程（主要是历史代码）
│  ├─ graphics_utils.py          # 投影、射线、相机矩阵等（PVPS 用到 project_numpy）
│  ├─ box_utils.py               # 3D bbox 角点/2D 边界 mask（PVPS 用到）
│  ├─ img_utils.py               # 可视化/辅助（PVPS 用到 visualize_depth_numpy 等）
│  ├─ colmap_utils.py, sh_utils.py, loss_utils.py, vq_utils.py, ...（NeRF 相关偏多）
│  └─ ...
│
├─ README.MD                    # 安装、下载与转换流程说明（以 NOTR/Street-NVS 为背景）
├─ requirements.txt / environment.yml
└─ LICENSE
```

要点：
- 与 Waymo PVPS 数据准备直接相关的核心目录是 `datasets/` 和 `datasets/pvps/`，以及 `utils/` 中少量通用投影/几何工具。
- `deeplab2/` 提供语义分割模型代码，可作为“预测语义图”的方案；若要使用 Waymo 官方“相机语义标注 GT”，需在转换阶段解析 TFRecord 中的 camera segmentation labels（当前仓库未提供现成脚本，见第 5 节建议）。
- `utils/` 下大量 NeRF/Colmap/渲染相关工具为历史遗留，PVPS 数据准备无需依赖（详见第 2.B 节与第 5 节）。


## 2. 核心功能识别与代码定位

### A. Waymo PVPS 数据准备相关代码

满足“选择场景 + 提取 RGB/语义/深度”目标的现有组件：

- 场景筛选（选择含相机语义标注的片段）
  - `datasets/pvps/WOD/2d_pvps_*.txt`：Waymo 官方整理的 PVPS 有标注帧清单。
  - `datasets/pvps/pvps_count.py`：从上述帧清单聚合到“每个 segment 的带标注帧数”，输出到 `datasets/pvps/pvps_stats/*.txt`。
  - `datasets/pvps/find_segment_indices.py`：把“带标注的 segment 名”映射到 `data/waymo_{train,valid}_list.txt` 的行号（0-index），方便后续按索引下载/转换。

- 原始数据下载（TFRecord）
  - `datasets/download_waymo.py`
    - 输入：`--set {train|valid}`、`--scene_ids ...` 或 `--split_file data/waymo_splits/*.txt`；
    - 动作：并发调用 `gsutil cp -n` 下载 `{source}/{segment}.tfrecord` 至 `--target_dir`；
    - 产出：对应 split 的 `.tfrecord` 文件集。

- TFRecord 解析与序列落盘（图像/标定/位姿/点云投影/动态掩码等）
  - `datasets/waymo_converter.py`
    - 关键函数：`parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, ...)`
    - 支持处理项：`pose`（ego 与相机位姿）、`calib`（intrinsics/extrinsics）、`image`（RGB 图）、`lidar`（聚合五把 LiDAR 的 3D 点与到多相机的投影像素，保存为 pointcloud.npz）、`track`（基于 GT 或外部跟踪器的 3D 框与可见性统计）、`dynamic_mask`（>1m/s 的运动体投影得到二值掩码）。
    - 产出：`processed/{split}/{scene_id}/images|intrinsics|extrinsics|ego_pose|pointcloud.npz|dynamic_mask|track/`。

- 从投影点云生成稀疏 LiDAR 深度图
  - `datasets/generate_lidar_depth.py`（单序列版）与 `datasets/pvps/generate_depth.py`（批量遍历序列版）
    - 输入：`processed/.../{scene}/images` 与 `pointcloud.npz`、`intrinsics/`、`extrinsics/`；
    - 动作：筛出投影到目标相机的 LiDAR 点，按像素坐标取最小深度，得到稀疏深度；可视化存 `png`（默认仅 cam 0）。
    - 产出：`processed/.../{scene}/lidar_depth/{frame_cam}.npy + {frame_cam}.png`。

与“语义图”最相关但当前缺失的环节：
- 读取 Waymo TFRecord 中的 camera segmentation labels 并导出每帧语义图。仓库里有 `datasets/pvps/WOD/camera_segmentation.proto` 但 `waymo_converter.py` 尚未解析相机语义标签。可在该脚本中新增 `semantic` 处理分支，使用官方 proto 将 `frame.camera_labels`（或相机语义字段）解码后落盘为 PNG/NPY（第 5 节提供具体改造建议）。
- 替代方案：使用 `deeplab2/` 对 RGB 做推理生成语义图（非 GT）。


### B. NeRF 相关历史代码（待清理候选）

以下模块/文件主要服务于旧的 NeRF/Street-NVS/Colmap 流程，和 PVPS 数据准备无直接关系：

- `utils/waymo_utils.py`：大量关于 COLMAP、对象轨迹、点云融合、背景球体、sky mask 等逻辑；含 `generate_dataparser_outputs(...)` 等，为 NeRF/动态场景建模准备数据。
- `utils/colmap_utils.py`：读取/解析 COLMAP 二进制/文本模型、DB 写入等工具。
- `utils/graphics_utils.py`：包含投影/射线/球体相交/相机矩阵等，PVPS 仅用到其中 `project_numpy` 等少量函数；其余大部分为渲染/射线生成工具。
- `utils/sh_utils.py`：球谐（Spherical Harmonics）相关，典型 NeRF 组件。
- `utils/loss_utils.py`, `utils/vq_utils.py`, `utils/general_utils.py` 等：训练/渲染/特征量化相关。
- `deeplab2/src/...` 在本项目中主要用于训练/推理（若要使用 GT 语义则不需要）。
- `datasets/base/*`：抽象的数据源/数据集封装，更贴近训练管线；PVPS 数据准备阶段可以不依赖。

以上皆是重构时的“降噪”重点，可移动到 `legacy/` 或删除（若确认无依赖）。


## 3. 关键脚本和函数详解（PVPS 视角）

以下 4 个文件是当前 PVPS 数据准备最核心的组成：

### 3.1 `datasets/download_waymo.py`
- 目的：并发从 Google Cloud Storage 下载指定 split 的 `.tfrecord`。
- 主要函数：
  - `download_file(filename, target_dir, source)`：单文件 `gsutil cp -n` 下载；失败抛错。
  - `download_files(file_names, target_dir, source)`：线程池并发下载，打印进度。
- CLI 输入/输出：
  - 输入：`--set {train|valid}`，`--target_dir`，`--scene_ids ...` 或 `--split_file data/waymo_splits/*.txt`。
  - 输出：`target_dir` 下的 `{segment}.tfrecord` 文件。

### 3.2 `datasets/waymo_converter.py`
- 目的：将 TFRecord 解析为统一的序列文件夹结构，产出图像、内外参、ego 位姿、点云投影、跟踪信息与动态掩码等。
- 关键函数：
  - `get_extrinsic(camera_calibration)`：相机外参（Waymo vehicle 坐标到 OpenCV 右/下/前坐标）
  - `get_intrinsic(camera_calibration)`：相机内参 K
  - `project_label_to_image(dim, obj_pose, calibration)`：3D 盒角点投影到指定相机，返回像素坐标与可见性掩码
  - `project_label_to_mask(dim, obj_pose, calibration)`：3D 盒在像素平面的 2D polygon mask
  - `parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, ...)`：主处理逻辑；按 `process_list` 执行不同子任务：
    - `pose`：保存 `ego_pose/{frame}.txt` 与相机 pose `ego_pose/{frame}_{cam}.txt`，并输出 `timestamps.json`
    - `calib`：保存 `intrinsics/{cam}.txt` 与 `extrinsics/{cam}.txt`
    - `image`：保存 `images/{frame_cam}.png`
    - `lidar`：聚合五把 LiDAR 点云与投影，保存 `pointcloud.npz`（含 `pointcloud` 和 `camera_projection` 两个字典）
    - `track`：保存 `track/` 下的 `track_info.txt`、`track_camera_vis.json`、可视化视频等
    - `dynamic_mask`：保存 `dynamic_mask/{frame_cam}.png`
  - `main()`：读取 `--split_file`（scene_id 与 seq_name 对齐校验）、`--segment_file`（master list），逐个 scene 调用 `parse_seq_rawdata(...)`。
- 典型输入/输出：
  - 输入：`--root_dir` 指向 `raw/{split}`，`--save_dir` 指向 `processed/{split}`，split/scene 列表文件。
  - 输出：`processed/{split}/{scene_id}/` 目录树（含 images/intrinsics/extrinsics/ego_pose/pointcloud.npz/...）。

### 3.3 `datasets/generate_lidar_depth.py`
- 目的：针对“单个序列”将 `pointcloud.npz` 投影到 `images/` 上生成“稀疏 LiDAR depth”。
- 主要函数：
  - `load_calibration(datadir)`：加载 `intrinsics/` 与 `extrinsics/`（注意：原脚本中有一次重复读取 extrinsics 的小问题，`datasets/pvps/generate_depth.py` 已修正）
  - `generate_lidar_depth(datadir)`：核心逻辑；逐帧筛选投影到相机的 LiDAR 点，对像素坐标聚合最小深度，保存 `npy` 与可视化 `png`。
- 输入/输出：
  - 输入：`--datadir processed/{split}/{scene_id}`
  - 输出：`lidar_depth/{frame_cam}.npy`（包含 mask+value）与 `lidar_depth/{frame_cam}.png`（可视化，默认 cam 0）。

### 3.4 `datasets/pvps/generate_depth.py`
- 目的：对“某个 split 的全部序列”批量生成稀疏 LiDAR depth（对 3.3 的批量包装）。
- 主要函数：
  - `process_single_sequence(seq_dir)`：几乎等同 3.3 的 `generate_lidar_depth`；
  - `main()`：遍历 `--datadir processed/{split}` 下的子目录（按数字升序），逐个调用 `process_single_sequence`。
- 输入/输出：
  - 输入：`--datadir processed/{split}`
  - 输出：各序列下的 `lidar_depth/`。


（辅助：`utils/box_utils.py` 的 `get_bound_2d_mask` 与 `bbox_to_corner3d` 在 `waymo_converter.py` 的投影与动态掩码生成中被频繁使用；`utils/graphics_utils.py` 的 `project_numpy` 用于 3D->2D 投影。）


## 4. 数据流分析（端到端）

推荐的 PVPS 数据准备流水线如下：

```
输入：Waymo Open Dataset TFRecord（含相机图像、LiDAR、可选相机语义标签）

→ 步骤 A：场景筛选 / 列表生成（按“有相机语义标注”的 segment 过滤）
  - 使用 datasets/pvps/WOD/2d_pvps_*.txt（Waymo 官方帧级清单）
  - 运行 datasets/pvps/pvps_count.py 统计到 segment 级别
  - 运行 datasets/pvps/find_segment_indices.py 将 segment 名映射到 data/waymo_{train,valid}_list.txt 中的索引

→ 步骤 B：下载原始数据（TFRecord）
  - 运行 datasets/download_waymo.py --set {train|valid} --scene_ids ...（或 --split_file ...）

→ 步骤 C：TFRecord 转换为文件夹结构
  - 运行 datasets/waymo_converter.py 
    产出 processed/{split}/{scene_id}/
      ├─ images/{frame_cam}.png
      ├─ intrinsics/{cam}.txt, extrinsics/{cam}.txt
      ├─ ego_pose/{frame}.txt (+ 每相机 pose)
      ├─ pointcloud.npz（聚合 3D 点与投影信息）
      ├─ dynamic_mask/{frame_cam}.png（可选）
      └─ track/*（可选）

→ 步骤 D：生成 LiDAR 深度图
  - 单序列：datasets/generate_lidar_depth.py --datadir processed/{split}/{scene_id}
  - 批量：datasets/pvps/generate_depth.py --datadir processed/{split}

→ 步骤 E：生成语义图（两种路线，任选其一）
  - E1（推荐/GT）：扩展 waymo_converter.py 解析 TFRecord 中的 camera segmentation labels，直接导出每帧语义 PNG/NPY；
  - E2（预测）：使用 deeplab2/ 对 images/ 做推理，保存 predicted semantics。

输出：
  - RGB 街景图：processed/{split}/{scene_id}/images/{frame_cam}.png
  - LiDAR 深度图：processed/{split}/{scene_id}/lidar_depth/{frame_cam}.npy/.png
  - 语义图：processed/{split}/{scene_id}/semantics/{frame_cam}.png（需按 E1 或 E2 生成）
```


## 5. 重构建议（可操作清单）

为聚焦 PVPS 数据准备，建议按“保留最小闭环 + 逐步裁剪”的原则实施：

1) 优先补齐“语义图导出”能力（核心增量）
- 在 `datasets/waymo_converter.py` 新增 `semantic` 处理分支：
  - 使用 `simple-waymo-open-dataset-reader` 或 Waymo 官方 API 读取 camera segmentation labels（仓库内已含 `datasets/pvps/WOD/camera_segmentation.proto` 可作为参考）；
  - 为每个 `frame × camera` 解析出语义 label map，并落盘到 `processed/{split}/{scene}/semantics/{frame_cam}.png`；
  - 推荐使用调色板或保持 uint16/uint8 label 编码，便于下游训练/评测。
- 如果临时采用预测语义：在 `deeplab2/` 下提供一个简洁的推理脚本，用 `images/` 作为输入、输出 `semantics/` 目录。

2) 合并/统一 LiDAR 深度生成脚本（去重）
- `datasets/generate_lidar_depth.py` 与 `datasets/pvps/generate_depth.py` 功能高度重叠：
  - 建议保留一个脚本，增加参数 `--mode {single,batch}` 或直接检测传入路径是“序列目录”还是“split 目录”；
  - 采用 `datasets/pvps/generate_depth.py` 中已修正的 `load_calibration` 实现，避免 extrinsics 重复读取的小瑕疵。

3) 提取最小通用几何工具，隔离 NeRF 依赖
- 当前 `utils/graphics_utils.py` 体量较大，PVPS 仅需 `project_numpy` 等少量函数：
  - 将投影/相机矩阵等“与渲染无关”的小函数抽到 `utils/pvps_geom.py`（新文件），
  - `waymo_converter.py`、`generate_lidar_depth.py` 等改为只依赖该轻量模块，便于删除其他 NeRF 逻辑。

4) 移动/删除明显的 NeRF 遗留代码（在确认无依赖后）
- 可移动到 `legacy/` 或直接移除的候选：
  - `utils/waymo_utils.py`（NeRF 数据解析与 COLMAP/点云流程）
  - `utils/colmap_utils.py`、`utils/sh_utils.py`、`utils/loss_utils.py`、`utils/vq_utils.py`、`utils/general_utils.py` 等
  - `datasets/base/*`（如仅数据准备使用，可暂存 `legacy/`）
  - `deeplab2/src/...`（若不走预测语义路线，可暂存）
- 注意：`utils/box_utils.py` 与 `utils/img_utils.py` 存在真实被 PVPS 使用的函数，应保留。

5) 默认配置与 README 对齐 PVPS 目标
- 将 `datasets/waymo_converter.py` 的 `--process_list` 默认值调整为 `['pose','calib','image','lidar']`（去掉 `track` 与 `dynamic_mask`），
  同时在 README 中给出“最小 PVPS 数据准备”命令示例，另以选项方式开启 `track/dynamic_mask/semantic`。

6) 健壮性与可维护性
- 在 `waymo_converter.py` 的 `image` 分支中统一输出后缀（png/jpg）并去重；
- 对 `pointcloud.npz` 的结构写入一个轻量的 schema 注释（keys/shape/含义），减少重复阅读代码；
- 对分辨率/相机顺序等固定约定（5 个相机的 cam-id 与宽高）在 utils 中集中定义常量；
- 为 `datasets/pvps/find_segment_indices.py` 去掉硬编码路径，使用 CLI 参数与相对路径默认值。


## 附：与 PVPS 直接相关的函数速览（输入/输出）

- `datasets/waymo_converter.py::parse_seq_rawdata(...)`
  - 输入：
    - `process_list: List[str]`（['pose','calib','image','lidar',...]）
    - `root_dir: str`（raw/{split}）
    - `seq_name: str`（segment 文件名）
    - `seq_save_dir: str`（processed/{split}/{scene_id}）
  - 输出：在 `seq_save_dir` 下生成 `images/`、`intrinsics/`、`extrinsics/`、`ego_pose/`、`pointcloud.npz` 等。

- `datasets/generate_lidar_depth.py::generate_lidar_depth(datadir)`
  - 输入：`datadir=processed/{split}/{scene_id}`
  - 输出：`lidar_depth/{frame_cam}.npy`（{'mask': H×W bool, 'value': N float}）与 `lidar_depth/{frame_cam}.png` 可视化。

- `datasets/pvps/pvps_count.py::analyze_and_save_stats(input_filepath, output_filepath)`
  - 输入：帧级清单 `2d_pvps_*.txt`（格式：`segment_name,frame_idx`）
  - 输出：`{segment_name,count}` 文本；并返回总帧数。

- `datasets/pvps/find_segment_indices.py::find_indices_in_master_list(master_list_path, annotated_segments_path)`
  - 输入：master 列表（`waymo_train_list.txt` 或 `waymo_valid_list.txt`）与“带标注 segment 清单”（上一步输出）
  - 输出：这些 segment 在 master 列表中的“行号索引”（0-index），可直接交给 `download_waymo.py`/`waymo_converter.py` 使用。


---

完成度与下一步：
- 当前代码已完整覆盖：下载（TFRecord）→ 解析（图像/标定/位姿/点云）→ LiDAR 深度图生成；
- 尚待补齐：从 TFRecord 直接导出“相机语义 GT”。建议优先在 `waymo_converter.py` 增加 `semantic` 分支，复用 `camera_segmentation.proto`，实现一键导出。

如需我基于以上建议直接提交改造 PR（新增 `semantic` 落盘与合并深度脚本），请告知优先级与目标输出格式（PNG 调色板/uint16 标签/NPY）。
