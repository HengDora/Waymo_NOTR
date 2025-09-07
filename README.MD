# Preparing NeRF On-The-Road (NOTR) Dataset

## 0. Installation

```
# Set conda environment
conda create -n waymo_NOTR python=3.8
conda activate waymo_NOTR

# Install torch (corresponding to your CUDA version)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements
pip install -r requirements.txt

# Install submodules
pip install ./submodules/simple-waymo-open-dataset-reader
```

## 1. 注册 Waymo Open Dataset
### 注册 Waymo Open Dataset 账户并安装 gcloud SDK
要下载 Waymo 数据集，您需要在 [Waymo Open Dataset](https://waymo.com/open/) 注册一个账户。同时，您需要安装 gcloud SDK 并完成账户认证。请参考 [此页面](https://cloud.google.com/sdk/docs/install) 获取安装和设置 gcloud SDK 的详细信息。

运行以下命令进行认证：
```bash
gcloud auth login

### Set Up the Data Directory

Once you've registered and installed the gcloud SDK, create a directory to house the raw data:

# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/waymo/raw   
mkdir -p ./data/waymo/processed 
```

## 2. 下载原始数据

使用提供的 Python 脚本 datasets/download_waymo.py 开始下载所需的数据样本。

### 从 Waymo Open Dataset 下载特定场景

您可以通过指定 --set 参数从 train 或 valid 数据集下载特定场景，并结合 --scene_ids 参数指定场景ID。

例如，从训练集下载第 2、31 和 36 个场景：
```shell
python datasets/download_waymo.py \
    --set train \
    --target_dir ./data/waymo/raw/training \
    --scene_ids 2 31 36
```

从验证集下载相同的场景：
```
python datasets/download_waymo.py \
    --set valid \
    --target_dir ./data/waymo/raw/validation \
    --scene_ids 6 26 90 105 108 134 150 181
```

### 使用 Split 文件下载不同数据拆分

<details>
<summary>点击展开详细说明</summary>

您还可以使用 --split_file 参数下载 split 文件中指定的场景。脚本将从文件中读取场景ID，并从指定的数据集（train 或 valid）下载对应场景。

例如，从训练集下载自定义 split 文件中的场景：
- **Static32 Split:**

```shell
python datasets/download_waymo.py  --set train --split_file data/waymo_splits/static32.txt
```

- **Dynamic32 Split:**

```shell
python datasets/download_waymo.py --set train --split_file data/waymo_splits/dynamic32.txt
```

- **Diverse56 Split:**

```shell
python datasets/download_waymo.py --set train --split_file data/waymo_splits/diverse56.txt
```

请确保根据您的项目目录结构和需求修改路径和文件名。脚本需要 data/waymo_train_list.txt（训练集）和 data/waymo_valid_list.txt（验证集）这两个文件，其中包含场景名称列表。

### 数据集拆分

对于 Waymo Open Dataset，场景名称按字母顺序排列并存储在 data/waymo_train_list.txt（训练集）和 data/waymo_valid_list.txt（验证集）中。场景索引由行号减一确定（从 0 开始计数）：

**Static-32**: 3, 19, 36, 69, 81, 126, 139, 140, 146, 148, 157, 181, 200, 204, 226, 232, 237, 241, 245, 246, 271, 297, 302, 312, 314, 362, 482, 495, 524, 527, 753, 780

**Dynamic-32**: 16, 21, 22, 25, 31, 34, 35, 49, 53, 80, 84, 86, 89, 94, 96, 102, 111, 222, 323, 323, 382, 382, 402, 402, 427, 427, 438, 438, 546, 581, 592, 620, 640, 700, 754, 795, 796

**Diverse-56**:

- Ego-static: 1, 23, 24, 37, 66, 108, 114, 115
- Dusk/Dawn: 124, 147, 206, 213, 574, 680, 696, 737
- Gloomy: 47, 205, 220, 284, 333, 537, 699, 749
- Exposure mismatch: 58, 93, 143, 505, 545, 585, 765, 766
- Nighttime: 7, 15, 30, 51, 130, 133, 159, 770
- Rainy: 44, 56, 244, 449, 688, 690, 736, 738
- High-speed: 2, 41, 46, 62, 71, 73, 82, 83
</details> 


### 脚本会从指定的 Google Cloud Storage 存储桶下载 .tfrecord 文件：

如需更多信息或创建自定义拆分，请参考 data/waymo_splits/ 目录以及列表文件（data/waymo_train_list.txt 和 data/waymo_valid_list.txt）。

## 3. 数据预处理
使用 script/waymo/waymo_converter.py 脚本处理 Waymo 数据。以下是针对训练集和验证集的示例用法。

处理 demo.txt 中定义的场景(--set train 2 31 36)：

```
python ./datasets/waymo_converter.py \
    --root_dir ./data/waymo/raw/training \
    --save_dir ./data/waymo/processed/training \
    --split_file ./data/waymo_splits/demo.txt \
    --segment_file ./data/waymo_train_list.txt
```

处理 val_dynamic.txt 中定义的场景(--set valid 6 26 90 105 108 134 150 181)：

```
python ./datasets/waymo_converter.py \
    --root_dir ./data/waymo/raw/validation \
    --save_dir ./data/waymo/processed/validation \
    --split_file ./data/waymo_splits/val_dynamic.txt \
    --segment_file ./data/waymo_valid_list.txt
```

生成 LiDAR depth
```
python ./datasets/generate_lidar_depth.py --datadir DATA_DIR
```
这里的DATA_DIR是processed后的路径。
## 4.处理后数据介绍
经过预处理后，Waymo Open Dataset 将被组织为以下组件：

- **图像 (Images)**  
  - 所有帧图像以 `{timestep:03d}_{cam_id}.jpg` 命名，其中 `timestep` 为三位数字的时间步长，`cam_id` 为相机ID，分别为：
    - 0：FRONT（前置相机）
    - 1：FRONT_LEFT（前左相机）
    - 2：FRONT_RIGHT（前右相机）
    - 3：SIDE_LEFT（侧左相机）
    - 4：SIDE_RIGHT（侧右相机）

- **自车位姿 (Ego Poses)**  
  - 每个文件以 `{timestep:03d}.txt` 命名，包含一个 4x4 的自车到世界坐标系的变换矩阵。

- **相机内参 (Camera Intrinsics)**  
  - 每个文件以 `{cam_id}.txt` 命名，包含一个一维数组 `[f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]`，表示相机的焦距、主点坐标和畸变参数。

- **相机外参 (Camera Extrinsics)**  
  - 每个文件以 `{cam_id}.txt` 命名，包含一个 4x4 的相机到自车坐标系的变换矩阵，即 Waymo Open Dataset 中的 `frame.context.camera_calibrations.extrinsic.transform`。

- **LiDAR_depth 数据 (Lidar Data)**  

- **动态掩码 (Dynamic Mask)**  
  - 二值掩码图像以 `{timestep:03d}_{cam_id}.png` 命名，用于指示场景中的动态区域。1 表示动态，0 表示静态。通过过滤地面真值 2D 对象边界框获得，排除速度小于 1m/s 的边界框，以包含有意义的移动对象，同时减少背景噪声。用于评估，不在训练中使用。

### 目录结构
处理后的数据集将遵循以下目录结构：
```
data/waymo/processed
├── training
│   ├── SCENE_ID               # 场景ID，例如 001、002 等
│   │   ├── dynamic_masks      # 动态掩码：{timestep:03d}{cam_id}.png
│   │   ├── ego_pose           # 自车位姿：{timestep:03d}.txt
│   │   ├── extrinsics         # 相机外参：{cam_id}.txt
│   │   ├── images             # 图像：{timestep:03d}{cam_id}.jpg
│   │   ├── intrinsics         # 相机内参：{cam_id}.txt
│   │   ├── lidar_depth        # LiDAR 数据和对齐后的images
```