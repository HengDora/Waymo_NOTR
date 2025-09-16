#!/bin/bash

# ========================== 配置区域 ==========================
# 1. 设置包含Waymo图像的文件夹路径
IMAGE_DIR="/home/datuwsl/Research/SYSU/data/Waymo_NOTR/data/waymo/processed/validation/001/images"

# 2. 设置输出视频的文件名（视频会保存在IMAGE_DIR的上一级目录中）
OUTPUT_VIDEO="waymo_panorama_video.mp4"

# 3. 设置视频的帧率 (Hz), 根据您的数据修改，比如 10 或 5
FRAME_RATE=10
# ============================================================


# --- 脚本主体，无需修改 ---

if ! command -v ffmpeg &> /dev/null; then
    echo "错误: ffmpeg 未安装，请先安装 ffmpeg。"
    exit 1
fi

cd "$IMAGE_DIR" || { echo "错误: 无法进入目录 $IMAGE_DIR"; exit 1; }

TEMP_DIR="stitched_frames_temp"
mkdir -p "$TEMP_DIR"

echo "步骤 1/3: 准备拼接每一帧的图像..."

timestamps=$(ls -v *_0.png | sed 's/_0.png//')

if [ -z "$timestamps" ]; then
    echo "错误: 在目录 $IMAGE_DIR 中没有找到格式为 'timestamp_0.png' 的文件。"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# --- 进度条变量初始化 ---
counter=0
# 使用 wc -l (word count - lines) 来获取总行数，即总时间戳数量
total_files=$(echo "$timestamps" | wc -l)
echo "发现 $total_files 个时间戳，开始处理..."

# 循环处理每一个时间戳
for ts in $timestamps; do
  # --- 进度条更新 ---
  counter=$((counter + 1))
  percent=$((counter * 100 / total_files))
  # 构建进度条的可视化部分
  bar=$(printf "%0.s#" $(seq 1 $((percent / 2))))
  bar_padding=$(printf "%0.s-" $(seq 1 $((50 - (percent / 2)))))
  # 使用 \r 回到行首来刷新进度条，不换行
  printf "\r拼接中: [%s%s] %d%% (%d/%d)" "$bar" "$bar_padding" "$percent" "$counter" "$total_files"

  # --- 核心FFmpeg拼接命令 ---
  ffmpeg -y -hide_banner -loglevel error \
    -i "${ts}_3.png" -i "${ts}_1.png" -i "${ts}_0.png" -i "${ts}_2.png" -i "${ts}_4.png" \
    -filter_complex "[0:v]pad=w=1920:h=1280:x=0:y=(1280-ih)/2:color=black[p0];[4:v]pad=w=1920:h=1280:x=0:y=(1280-ih)/2:color=black[p4];[p0][1:v][2:v][3:v][p4]hstack=inputs=5[v]" \
    -map "[v]" "$TEMP_DIR/${ts}_stitched.png"
done

# --- 结束进度条并换行 ---
echo -e "\n步骤 1 完成！"


echo "步骤 2/3: 所有图像拼接完成，开始生成视频..."

if [ -z "$(ls -A $TEMP_DIR)" ]; then
    echo "错误: 未能成功生成任何拼接图像，视频创建中止。"
    exit 1
fi

ffmpeg -y -hide_banner -loglevel warning \
  -framerate $FRAME_RATE \
  -pattern_type glob -i "$TEMP_DIR/*_stitched.png" \
  -c:v libx264 -preset ultrafast -crf 18 \
  -pix_fmt yuv420p \
  "../${OUTPUT_VIDEO}"

if [ $? -ne 0 ]; then
    echo "错误: ffmpeg 创建视频失败。"
    echo "为方便调试，临时文件已保留在目录: $PWD/$TEMP_DIR"
    exit 1
fi

rm -rf "$TEMP_DIR"

echo "步骤 3/3: 视频制作完成并已清理临时文件！"
echo "视频已保存至: $(dirname "$PWD")/${OUTPUT_VIDEO}"