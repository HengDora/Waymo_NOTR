import sys
import os
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# 将工具函数放在脚本开头
# A small helper function to reliably get the numeric part of a directory name for sorting
def get_numeric_part(name):
    try:
        return int(''.join(filter(str.isdigit, name)))
    except ValueError:
        return -1

# Assuming this utility is available in your project structure
# If not, you might need to copy its implementation here.
# from utils.img_utils import visualize_depth_numpy
# For now, let's create a placeholder so the script can run
def visualize_depth_numpy(depth, min_d=0.0, max_d=80.0):
    """
    Visualizes a numpy depth map.
    Args:
        depth (numpy.ndarray): Depth map of shape (H, W).
        min_d (float): Minimum depth for color mapping.
        max_d (float): Maximum depth for color mapping.
    Returns:
        numpy.ndarray: A color-mapped visualization of the depth map.
        numpy.ndarray: The colormap used.
    """
    colormap = cv2.applyColorMap(
        cv2.convertScaleAbs((depth - min_d) / (max_d - min_d) * 255, alpha=255),
        cv2.COLORMAP_JET
    )
    # Set invalid depth pixels (where depth is 0 or less) to black
    colormap[depth <= 0] = 0
    return colormap, None # Return None for the second value to match original call signature

# --- 原有的核心逻辑函数 ---
# 这些函数处理单个序列，我们将其保留并直接调用

image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    
    intrinsics = []
    extrinsics = []
    # This loop seems to have a copy-paste error in the original code.
    # It loads extrinsics twice. Correcting it to load once.
    for i in range(5):
        # Load Intrinsics
        intrinsic_data = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic_data[0], intrinsic_data[1], intrinsic_data[2], intrinsic_data[3]
        intrinsic_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic_mat)
        
        # Load Extrinsics
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)
        
    return extrinsics, intrinsics

def process_single_sequence(seq_dir):
    """
    This function contains the original logic to process one sequence directory.
    It takes the path to a single sequence (e.g., './data/waymo/processed/validation/001').
    """
    print(f"Generating LiDAR depth for sequence: {seq_dir}")
    
    save_dir = os.path.join(seq_dir, 'lidar_depth')
    os.makedirs(save_dir, exist_ok=True)
    
    image_dir = os.path.join(seq_dir, 'images')
    image_files = sorted(glob(os.path.join(image_dir, "*.png")))
    image_files += sorted(glob(os.path.join(image_dir, "*.jpg")))
    
    if not image_files:
        print(f"Warning: No images found in {image_dir}. Skipping.")
        return

    pointcloud_path = os.path.join(seq_dir, 'pointcloud.npz')
    if not os.path.exists(pointcloud_path):
        print(f"Error: pointcloud.npz not found in {seq_dir}. Skipping.")
        return
        
    pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
    pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  
    
    extrinsics, intrinsics = load_calibration(seq_dir)
    
    for image_filename in tqdm(image_files, desc=f"Processing {os.path.basename(seq_dir)}"):
        image = cv2.imread(image_filename)
        h, w = image.shape[:2]
        
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        
        depth_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.npy')
        depth_vis_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.png')
        
        raw_3d = pts3d_dict.get(frame)
        raw_2d = pts2d_dict.get(frame)

        if raw_3d is None or raw_2d is None:
            # print(f"Warning: Missing point cloud data for frame {frame} in {seq_dir}. Skipping frame.")
            continue
            
        num_pts = raw_3d.shape[0]
        pts_idx = np.arange(num_pts)
        pts_idx = np.tile(pts_idx[..., None], (1, 2)).reshape(-1) # (num_pts * 2)
        raw_2d = raw_2d.reshape(-1, 3) # (num_pts * 2, 3)
        mask = (raw_2d[:, 0] == cam)
        
        points_xyz = raw_3d[pts_idx[mask]]
        points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)

        c2w = extrinsics[cam]
        w2c = np.linalg.inv(c2w)
        points_xyz_cam = points_xyz @ w2c.T
        points_depth = points_xyz_cam[..., 2]

        valid_mask = points_depth > 0.
        
        points_xyz_pixel = raw_2d[mask][:, 1:3]
        points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
        points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
        points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)
        
        depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
        u, v = points_coord[:, 0], points_coord[:, 1]
        indices = v * w + u
        np.minimum.at(depth, indices, points_depth[valid_mask])
        depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
        valid_depth_pixel = (depth != 0)
        valid_depth_value = depth[valid_depth_pixel]
        valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
                    
        depth_file = dict()
        depth_file['mask'] = valid_depth_pixel
        depth_file['value'] = valid_depth_value
        np.save(depth_path, depth_file)

        try:
            if cam == 0:
                depth_reshaped = depth.reshape(h, w).astype(np.float32)
                depth_vis, _ = visualize_depth_numpy(depth_reshaped)
                # Use BGR format for OpenCV
                depth_on_img = image.copy()
                depth_on_img[depth_reshaped > 0] = depth_vis[depth_reshaped > 0]
                cv2.imwrite(depth_vis_path, depth_on_img)
        except Exception as e:
            print(f'Error visualizing depth for {image_filename}: {e}')


# --- 新的主函数，用于批量处理 ---

def main():
    parser = argparse.ArgumentParser(description="Generate LiDAR depth maps for all sequences in a directory.")
    parser.add_argument('--datadir', required=True, type=str, 
                        help="Path to the parent directory containing processed sequence folders (e.g., './data/waymo/processed/validation').")
    args = parser.parse_args()

    parent_dir = args.datadir
    if not os.path.isdir(parent_dir):
        print(f"Error: Provided directory does not exist: {parent_dir}")
        return

    # Find all subdirectories that are likely sequence folders
    try:
        subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        # Sort directories numerically to process them in order
        subdirs.sort(key=get_numeric_part)
    except FileNotFoundError:
        print(f"Error: Cannot access directory contents: {parent_dir}")
        return

    print(f"Found {len(subdirs)} sequence(s) in '{parent_dir}': {subdirs}")
    
    for seq_name in subdirs:
        seq_path = os.path.join(parent_dir, seq_name)
        print("="*60)
        process_single_sequence(seq_path)
        print(f"Finished processing sequence: {seq_name}")
        print("="*60 + "\n")

    print("All sequences processed.")

if __name__ == "__main__":
    main()