import argparse
import os
from collections import Counter

def analyze_and_save_stats(input_filepath: str, output_filepath: str) -> int:
    """
    读取包含 'segment,frame' 数据的文件，统计每个segment的帧数，
    将结果保存到指定文件，并返回总帧数。

    Args:
        input_filepath (str): 包含原始数据的输入文件路径。
        output_filepath (str): 输出统计结果的文件路径。

    Returns:
        int: 该数据集中有标注的总帧数。
    """
    # 使用 Counter 来高效地统计每个 segment 的出现次数
    segment_counts = Counter()

    # 直接从输入文件路径读取数据
    with open(input_filepath, 'r') as f:
        for line in f:
            # 去除行首尾的空白字符
            clean_line = line.strip()
            if clean_line:
                # 按逗号分割，我们只需要第一部分 segment 名
                segment_name = clean_line.split(',')[0]
                segment_counts[segment_name] += 1

    # 将统计结果写入输出文件
    with open(output_filepath, 'w') as f:
        # 按照 segment 名排序，让输出文件更规整
        for segment, count in sorted(segment_counts.items()):
            # 写入格式为 "segment_name,count"
            f.write(f"{segment},{count}\n")

    # 计算并返回总帧数
    total_frames = sum(segment_counts.values())
    return total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从文件读取Waymo PVPS标注信息，统计每个segment的带标注帧数。"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        # 将你提供的路径设为默认值
        default="/home/datuwsl/Research/SYSU/data/waymo-open-dataset/tutorial",
        help="包含 2d_pvps_*.txt 文件的输入目录路径。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="用于保存输出的统计文件的目录路径。",
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义需要处理的数据集和对应的输入/输出文件名
    datasets_to_process = {
        "training": ("2d_pvps_training_frames.txt", "training_segment.txt"),
        "validation": ("2d_pvps_validation_frames.txt", "validation_segment.txt"),
        "test": ("2d_pvps_test_frames.txt", "test_segment.txt"),
    }

    total_counts = {}

    print(f"开始处理PVPS标注文件...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 30)


    for name, (input_filename, output_filename) in datasets_to_process.items():
        input_path = os.path.join(args.input_dir, input_filename)
        output_path = os.path.join(args.output_dir, output_filename)

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"!! 警告：输入文件不存在，已跳过: {input_path}")
            total_counts[name] = 0
            continue

        frame_count = analyze_and_save_stats(input_path, output_path)
        total_counts[name] = frame_count
        print(f"✔ 已分析 '{input_path}'\n  -> 生成 '{output_path}'")


    # ==============================================================================
    # 在命令行打印最终统计结果
    # ==============================================================================

    print("\n==================== PVPS 标注统计摘要 ====================")
    train_frames = total_counts.get("training", 0)
    valid_frames = total_counts.get("validation", 0)
    test_frames = total_counts.get("test", 0)

    print(f"训练集 (Training):   {train_frames:>7} 帧有标注")
    print(f"验证集 (Validation): {valid_frames:>7} 帧有标注")
    print(f"测试集 (Test):       {test_frames:>7} 帧有标注")
    print("---------------------------------------------------------")
    print(f"总计:              {train_frames + valid_frames + test_frames:>7} 帧有标注")
    print("=========================================================")