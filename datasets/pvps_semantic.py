import os
import argparse
from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from rich.console import Console
from rich.rule import Rule
import json
import re

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils

# 初始化 rich 控制台
console = Console()

# 如果 TF 没有开启 eager 模式，强制开启
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()


def process_waymo_tfrecord(tfrecord_path: str, output_dir: str, save_visualization: bool = True):
    """
    处理单个 Waymo PVPS tfrecord 文件，提取语义标签（以及可选的可视化图）。
    """
    camera_name_map = {
        "FRONT": "0",
        "FRONT_LEFT": "1",
        "FRONT_RIGHT": "2",
        "SIDE_LEFT": "3",
        "SIDE_RIGHT": "4",
    }

    filename = os.path.basename(tfrecord_path)
    console.print(Rule(f"[bold blue]正在处理: {filename}", style="blue"))

    semantic_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(semantic_labels_dir, exist_ok=True)

    if save_visualization:
        visualization_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")

    try:
        total_frames = sum(1 for _ in dataset)
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    except Exception as e:
        console.print(f"[yellow]警告：无法计算总帧数 ({e})，进度条将不显示总进度。[/yellow]")
        total_frames = None

    for data in tqdm(dataset, total=total_frames, desc="处理帧", unit="frame"):
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        if not frame.images[0].camera_segmentation_label.panoptic_label:
            continue

        for image_proto in frame.images:
            camera_name_str = open_dataset.CameraName.Name.Name(image_proto.name)
            if camera_name_str not in camera_name_map:
                continue

            timestamp = frame.timestamp_micros
            camera_code = camera_name_map[camera_name_str]
            base_filename = f"{timestamp}_{camera_code}"

            panoptic_label_proto = image_proto.camera_segmentation_label
            panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
                panoptic_label_proto
            )
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                panoptic_label, panoptic_label_proto.panoptic_label_divisor
            )
            semantic_label_2d = np.squeeze(semantic_label, axis=-1)

            label_path = os.path.join(semantic_labels_dir, f"{base_filename}.png")
            Image.fromarray(semantic_label_2d.astype(np.uint8)).save(label_path)

            if save_visualization:
                panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
                    semantic_label, instance_label
                )
                vis_path = os.path.join(visualization_dir, f"{base_filename}.png")
                Image.fromarray(panoptic_label_rgb).save(vis_path)

def rename_files_based_on_timestamps(output_dir: str, console: Console):
    """
    根据 timestamps.json 文件，将生成的时间戳文件名重命名为帧索引文件名。

    Args:
        output_dir (str): 当前 sequence 的输出目录 (例如 '.../validation/001')。
        console (Console): 用于打印信息的 rich Console 实例。
    """
    console.print(f"[cyan]开始重命名操作于: {output_dir}[/cyan]")
    
    timestamps_path = os.path.join(output_dir, "timestamps.json")
    if not os.path.exists(timestamps_path):
        console.print(f"[bold red]错误: 找不到 timestamps.json 文件于 {output_dir}，跳过重命名。[/bold red]")
        return

    # 1. 加载 timestamps.json 并构建 "时间戳 -> 帧索引" 的反向映射
    try:
        with open(timestamps_path, 'r') as f:
            timestamps_data = json.load(f)["FRAME"]
        
        # 将json中的浮点秒时间戳转换为微秒整型，用于匹配文件名
        timestamp_to_frame_map = {
            int(float(ts) * 1_000_000): frame_idx
            for frame_idx, ts in timestamps_data.items()
        }
    except Exception as e:
        console.print(f"[bold red]错误: 解析 timestamps.json 失败: {e}，跳过重命名。[/bold red]")
        return
        
    # 2. 定义需要重命名的子目录
    subdirs_to_rename = ["labels"]
    if os.path.exists(os.path.join(output_dir, "visualizations")):
        subdirs_to_rename.append("visualizations")

    total_renamed_count = 0
    # 3. 遍历子目录并执行重命名
    for subdir in subdirs_to_rename:
        current_dir = os.path.join(output_dir, subdir)
        if not os.path.isdir(current_dir):
            continue
            
        renamed_in_subdir = 0
        filenames = os.listdir(current_dir)
        # 使用tqdm来显示重命名进度
        for filename in tqdm(filenames, desc=f"重命名 {subdir}", unit="file"):
            if not filename.endswith(".png"):
                continue

            # 从文件名 '1553735853462203_0.png' 中提取时间戳和相机后缀
            match = re.match(r"(\d+)_(\d+)\.png", filename)
            if not match:
                continue

            timestamp_key = int(match.group(1))
            camera_suffix = f"_{match.group(2)}" # 保留相机后缀 '_0', '_1' 等

            # 在映射中查找对应的帧索引
            if timestamp_key in timestamp_to_frame_map:
                frame_index = timestamp_to_frame_map[timestamp_key]
                new_filename = f"{frame_index}{camera_suffix}.png"
                
                old_path = os.path.join(current_dir, filename)
                new_path = os.path.join(current_dir, new_filename)
                
                os.rename(old_path, new_path)
                renamed_in_subdir += 1
            else:
                # 只在第一次遇到警告时打印，避免刷屏
                if total_renamed_count == 0:
                     console.print(f"[yellow]警告: 在 {subdir}/{filename} 中找到的时间戳在 timestamps.json 中无匹配项。[/yellow]")
        
        console.print(f"在 [bold green]{subdir}[/bold green] 目录中成功重命名 {renamed_in_subdir} / {len(filenames)} 个文件。")
        total_renamed_count += renamed_in_subdir

    console.print(f"[bold cyan]重命名完成，共处理 {total_renamed_count} 个文件。[/bold cyan]")


def main():
    parser = argparse.ArgumentParser(description="根据场景ID列表提取 Waymo PVPS 标注信息并重命名")
    parser.add_argument(
        "--datadir", type=str, required=True,
        help="包含 .tfrecord 文件的根目录路径。"
    )
    parser.add_argument(
        "--outdir", type=str, required=True,
        help="保存处理结果的父目录路径。"
    )
    parser.add_argument(
        "--segment-file", type=str, required=True,
        help="包含所有segment名称列表的 .txt 文件路径。"
    )
    parser.add_argument(
        "--scene-ids", type=int, nargs='+', required=True,
        help="一个或多个要处理的场景索引 (0-based)，用空格分隔。"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="不保存彩色可视化图。"
    )
    args = parser.parse_args()

    console.print(Rule("[bold yellow]开始批量处理任务", style="yellow"))

    try:
        with open(args.segment_file, 'r') as f:
            all_segments = [line.strip() for line in f if line.strip()]
        console.print(f"成功从 '{os.path.basename(args.segment_file)}' 中读取 {len(all_segments)} 个segment名称。")
        console.print(f"计划处理 {len(args.scene_ids)} 个指定场景。")

        for scene_id in args.scene_ids:
            if scene_id >= len(all_segments):
                console.print(f"[bold red]警告: 索引 {scene_id} 超出文件列表范围 (共 {len(all_segments)} 行)，已跳过。[/bold red]")
                continue
            
            segment_name = all_segments[scene_id]
            input_tfrecord_path = os.path.join(args.datadir, f"{segment_name}.tfrecord")

            if not os.path.exists(input_tfrecord_path):
                console.print(f"[bold red]警告: 输入文件不存在: {input_tfrecord_path}，已跳过。[/bold red]")
                continue
            
            output_segment_dir = os.path.join(args.outdir, f"{scene_id:03d}")
            os.makedirs(output_segment_dir, exist_ok=True)
            
            # 步骤 1: 提取语义图
            process_waymo_tfrecord(
                tfrecord_path=input_tfrecord_path,
                output_dir=output_segment_dir,
                save_visualization=not args.no_vis
            )
            
            # 步骤 2: 立即进行重命名
            rename_files_based_on_timestamps(output_segment_dir, console)

    except FileNotFoundError:
        console.print(f"[bold red]错误: Segment列表文件未找到: {args.segment_file}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]处理过程中发生未知错误: {e}[/bold red]")

    console.print(Rule("[bold yellow]所有任务处理完毕", style="yellow"))


if __name__ == "__main__":
    main()