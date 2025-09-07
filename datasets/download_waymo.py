import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List


def download_file(filename, target_dir, source):
    result = subprocess.run(
        [
            "gsutil",
            "cp",
            "-n",
            f"{source}/{filename}.tfrecord",
            target_dir,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise Exception(result.stderr)


def download_files(
    file_names: List[str],
    target_dir: str,
    source: str,
) -> None:
    """
    Downloads a list of files from a given source to a target directory using multiple threads.

    Args:
        file_names (List[str]): A list of file names to download.
        target_dir (str): The target directory to save the downloaded files.
        source (str): The source directory to download the files from.
    """
    total_files = len(file_names)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_file, filename, target_dir, source)
            for filename in file_names
        ]

        for counter, future in enumerate(futures, start=1):
            try:
                future.result()
                print(f"[{counter}/{total_files}] Downloaded successfully!")
            except Exception as e:
                print(f"[{counter}/{total_files}] Failed to download. Error: {e}")


if __name__ == "__main__":
    print("note: `gcloud auth login` is required before running this script")
    print("Downloading Waymo dataset from Google Cloud Storage...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        type=str,
        choices=['train', 'valid'],
        default='valid',
        help="Dataset split to download (train or valid)"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/waymo/raw",
        help="Path to the target directory",
    )
    parser.add_argument(
        "--scene_ids", 
        type=int, 
        nargs="+", 
        help="scene ids to download"
    )
    parser.add_argument(
        "--split_file", 
        type=str, 
        default=None, 
        help="split file in data/waymo_splits"
    )
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)

    # 根据 --set 参数选择对应的 list 文件和 source 路径
    if args.set == 'train':
        total_list = open("data/waymo_train_list.txt", "r").readlines()
        source = "gs://waymo_open_dataset_v_1_4_3/individual_files/training"
    else:  # valid
        total_list = open("data/waymo_valid_list.txt", "r").readlines()
        source = "gs://waymo_open_dataset_v_1_4_3/individual_files/validation"

    # 获取文件列表
    if args.split_file is None:
        file_names = [total_list[i].strip() for i in args.scene_ids]
    else:
        split_file = open(args.split_file, "r").readlines()[1:]
        scene_ids = [int(line.strip().split(",")[0]) for line in split_file]
        file_names = [total_list[i].strip() for i in scene_ids]

    download_files(file_names, args.target_dir, source)