# Preparing NeRF On-The-Road (NOTR) Dataset

## 0. Installation

```
# Set conda environment
conda create -n waymo_NOTR python=3.9 -y
conda activate waymo_NOTR

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

例如，从**验证集**下载有pvps标注的场景：
```
python datasets/download_waymo.py \
    --set valid \
    --target_dir ./data/waymo/raw/validation \
    --scene_ids 1 12 30 44 46 92 98 108 140 143 150 155 165 167 174 178 180 188 195 197
```

<details>
<summary>从训练集下载有pvps标注的场景</summary>

```shell
python datasets/download_waymo.py \
    --set train \
    --target_dir ./data/waymo/raw/training \
    --scene_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 21 22 23 24 25 26 27 28 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 53 54 57 58 59 60 61 62 63 64 65 66 67 68 69 71 72 73 74 75 76 78 79 80 81 82 83 84 85 86 87 88 89 91 93 94 95 96 97 99 100 101 102 103 104 105 106 107 108 109 110 111 112 114 115 116 118 119 120 122 123 125 126 127 128 129 132 134 136 137 138 139 140 141 142 143 145 146 147 148 150 151 152 153 154 155 157 158 159 160 161 162 163 164 165 166 168 170 171 172 173 174 175 176 179 182 183 184 185 186 187 188 189 190 191 192 195 196 197 198 199 201 202 203 204 205 206 208 209 210 211 212 214 215 216 217 218 219 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 236 238 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 275 276 277 278 279 280 281 282 284 285 286 287 288 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 307 308 309 310 311 312 313 314 315 316 317 318 319 321 322 323 324 325 326 327 329 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 348 350 351 353 354 355 356 357 358 359 360 361 362 363 364 365 366 368 369 370 371 372 373 374 375 376 377 378 379 381 382 383 384 385 386 387 388 389 390 391 394 396 397 398 399 400 401 402 403 404 405 407 408 409 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 456 457 458 459 461 462 463 464 465 467 468 470 471 472 473 474 475 476 477 478 479 481 482 483 484 485 486 488 489 490 491 492 493 494 495 497 498 499 500 501 502 504 506 507 508 510 511 512 513 514 518 519 520 521 523 524 525 526 527 528 529 530 531 534 536 538 539 540 541 542 543 544 545 546 547 548 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 571 572 573 574 575 576 577 578 579 580 581 582 584 586 587 588 589 590 591 592 594 595 596 597 598 599 600 601 602 603 604 605 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 645 646 647 648 649 650 651 652 653 654 655 656 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 674 675 676 677 678 679 680 681 682 683 684 686 687 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 707 708 709 711 712 713 714 715 716 717 718 719 721 722 723 725 726 727 728 729 730 732 733 734 735 736 737 738 739 740 741 742 743 745 746 747 748 749 750 751 752 754 755 756 757 758 759 760 761 762 763 764 765 767 769 770 771 772 773 776 778 779 780 781 782 784 785 786 787 788 789 791 792 793 795 796 797
```

</details> 


### 脚本会从指定的 Google Cloud Storage 存储桶下载 .tfrecord 文件：

如需更多信息或创建自定义拆分，请参考 data/waymo_splits/ 目录以及列表文件（data/waymo_train_list.txt 和 data/waymo_valid_list.txt）。

## 3. 数据预处理
使用 script/waymo/waymo_converter.py 脚本处理 Waymo 数据。以下是针对训练集和验证集的示例用法。

处理 validation20.txt 中定义的场景，即有pvps标注的sequence (--set valid 1 12 30 44 46 92 98 108 140 143 150 155 165 167 174 178 180 188 195 197)：

```
python ./datasets/waymo_converter.py \
    --root_dir ./data/waymo/raw/validation \
    --save_dir ./data/waymo/processed/validation \
    --split_file ./data/waymo_splits/validation20.txt \
    --segment_file ./data/waymo_valid_list.txt
```

### 生成激光雷达深度图 (Generate LiDAR Depth Maps)

本脚本用于批量处理 **Waymo 数据集**，为指定父目录（如validation）下的每一个序列（sequence）文件夹生成稀疏的激光雷达深度图。

在终端中运行以下命令：

```shell
python generate_depth.py --datadir <path_to_parent_directory>

python ./datasets/generate_depth.py --datadir Waymo_NOTR/data/waymo/raw/validation
```
这里的DATA_DIR是processed后的路径。

--datadir

(必需) 指向包含所有已处理序列文件夹的 父目录路径。
脚本会自动遍历此目录下的所有子文件夹，并对每一个进行处理。

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


## 5.提取 Waymo PVPS 标注信息

本脚本用于批量处理 **Waymo 数据集**，从序列文件夹中的 `.tfrecord` 文件中提取 **语义分割标签**，并可选保存彩色可视化结果。

### 使用方法

在终端中运行以下命令：

```shell
python ./datasets/pvps_semantic.py \
  --datadir /home/hw08/project/zqy/DynamicStreet-4DGS/Waymo_NOTR/data/waymo/raw/validation \
  --outdir /home/hw08/project/zqy/DynamicStreet-4DGS/Waymo_NOTR/data/waymo/processed/validation \
  --segment-file /home/hw08/project/zqy/DynamicStreet-4DGS/Waymo_NOTR/data/waymo_valid_list.txt \
  --scene-ids 1 12 30 44 46 92 98 108 140 143 150 155 165 167 174 178 180 188 195 197
```

## 参数说明

- **--datadir**  
  *(必需)* 指向包含 **若干 `.tfrecord` 文件** 的目录路径。  
  脚本会自动遍历此目录下的所有 `.tfrecord` 文件，并逐一处理。  

- **--outdir**  
  *(必需)* 保存处理结果的父目录路径。  
  每个 `.tfrecord` 文件会在此目录下生成一个独立的输出文件夹。  

- **--no-vis**  
  *(可选)* 加上该参数后，不会保存彩色可视化的标签图，只保存灰度语义标签图。  

