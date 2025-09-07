import os
from typing import List

def find_indices_in_master_list(master_list_path: str, annotated_segments_path: str) -> List[int]:
    """
    在一个主列表文件中，查找带标注的segment列表，并返回它们的行号（0-indexed）。

    Args:
        master_list_path (str): 包含所有segment名称的主列表文件路径。
        annotated_segments_path (str): 包含带标注segment名称的CSV文件路径。

    Returns:
        List[int]: 找到的带标注segment在主列表中的索引号列表。
    """
    print(f"  - 正在读取总表文件: '{os.path.basename(master_list_path)}'")
    
    # 1. 读取主列表，并创建一个 segment_name -> index 的映射字典
    master_name_to_index = {}
    with open(master_list_path, 'r') as f:
        for i, line in enumerate(f):
            # 关键修正：同时去掉前缀 "segment-" 和后缀 "_with_camera_labels"
            clean_name = line.strip().replace('segment-', '').replace('_with_camera_labels', '')
            master_name_to_index[clean_name] = i
    
    print(f"    在总表中找到 {len(master_name_to_index)} 个 Segment。")

    # 2. 读取带标注的segment列表
    print(f"  - 正在读取带标注 Segment 文件: '{os.path.basename(annotated_segments_path)}'")
    annotated_names = set()
    with open(annotated_segments_path, 'r') as f:
        for line in f:
            # 文件格式是 "segment_name,count"，我们只需要 segment_name
            segment_name = line.strip().split(',')[0]
            annotated_names.add(segment_name)
    
    print(f"    找到 {len(annotated_names)} 个需要搜索的带标注 Segment。")

    # 3. 查找索引
    found_indices = []
    not_found_count = 0
    for name in annotated_names:
        if name in master_name_to_index:
            found_indices.append(master_name_to_index[name])
        else:
            not_found_count += 1
    
    if not_found_count > 0:
        print(f"    警告: 有 {not_found_count} 个带标注 Segment 未在总表中找到。")

    # 4. 对索引进行排序并返回
    return sorted(found_indices)


if __name__ == "__main__":
    # --- 文件路径硬编码区 ---
    # 请根据你的实际情况修改这些路径
    base_data_dir = "/home/datuwsl/Research/SYSU/data/Waymo_NOTR/data" # 假设一个基础目录
    
    hardcoded_paths = {
        'train_master': os.path.join(base_data_dir, 'waymo_train_list.txt'),
        'valid_master': os.path.join(base_data_dir, 'waymo_valid_list.txt'),
        'train_annotated': './pvps_stats/training_segment.txt', # 假设这是你上一步脚本的输出
        'valid_annotated': './pvps_stats/validation_segment.txt', # 假设这是你上一步脚本的输出
    }
    
    print("开始查找 Segment 索引...\n")
    
    # --- 处理训练集 ---
    print("--- 正在处理训练集 ---")
    try:
        train_indices = find_indices_in_master_list(
            hardcoded_paths['train_master'],
            hardcoded_paths['train_annotated']
        )
        print(f"\n=> 为 训练集 找到 {len(train_indices)} 个索引:")
        # 使用空格分割输出
        print("   " + " ".join(map(str, train_indices)))
        
    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 -> {e}。请检查文件路径。")
    
    print("\n" + "="*50 + "\n")
    
    # --- 处理验证集 ---
    print("--- 正在处理验证集 ---")
    try:
        valid_indices = find_indices_in_master_list(
            hardcoded_paths['valid_master'],
            hardcoded_paths['valid_annotated']
        )
        print(f"\n=> 为 验证集 找到 {len(valid_indices)} 个索引:")
        # 使用空格分割输出
        print("   " + " ".join(map(str, valid_indices)))

    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 -> {e}。请检查文件路径。")

    print("\n脚本运行结束。")