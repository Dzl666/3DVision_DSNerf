import numpy as np
import glob
from os.path import join, exists
from os import makedirs
import cv2

uint2float = lambda x : (x.astype(np.float32) / 255.)
axis_transform = np.linalg.inv(np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))

def convert(base_dir, testing_hold=8, depth_start=0):
    """testing_hold: one test image for every "testing_hold" images
        depth_start: not used, set this to zero (which depth map to start with)
    """
    
    train_images = []
    test_images = []
    train_poses = []
    test_poses = []
    train_depths = []
    max_bds = 0
    min_bds = np.inf

    intrinsics_path = join(base_dir, "Video/Intrinsics.txt")    # intrinsics of the video
    extrinsics_path = join(base_dir, "Video/Pose.txt")          # poses of the video
    depths_dir = join(base_dir, "depth_colmap_opt")                   # directory to the output of colmap optimization
    save_dir = join(base_dir, "llff_colmap")                    # where to save the outputs
    if not exists(save_dir):
        makedirs(save_dir)

    def create_key(p):
        return int(p.split("\\")[-1][3:-4])
    # sort depths to have the same order as rgb_paths
    depths_paths = sorted(glob.glob(join(depths_dir, "im_*.npy")), key=create_key)

    # get corresponding rgb image paths
    rgb_paths = np.load(join(depths_dir, "rgb_paths.npy"))

    # generate intrinsics
    intrinsics = np.loadtxt(intrinsics_path)[:9]
    # units: pixel
    focal = (int)(0.5 * (intrinsics[0] + intrinsics[4]))
    H = cv2.imread(rgb_paths[0]).shape[0]
    W = cv2.imread(rgb_paths[0]).shape[1]

    # get all video extrinsics
    extrinsics = np.loadtxt(extrinsics_path)[:, 1:].reshape(-1, 4, 4)

    for ind in range(len(depths_paths)):
        # load depth and RGB-[720, 1080, 3]
        rgb = uint2float(cv2.cvtColor(cv2.imread(rgb_paths[ind]), cv2.COLOR_BGR2RGB))
        print(f"loaded rgb: {str(rgb_paths[ind])}")
        # original unit: mm ?
        depth = 1e-3 * np.load(depths_paths[ind])
        nonzero_idx = np.nonzero(depth)
        depth_max = np.max(depth)
        depth_min = np.min(depth[nonzero_idx])
        if(depth_max > max_bds):
            max_bds = depth_max
        if(depth_min < min_bds):
            min_bds = depth_min
        print(f"loaded depth: {str(depths_paths[ind])}")

        # get current row of intrinsics and concat with HWF
        rgb_row = int(rgb_paths[ind].split("\\")[-1][:-4])
        print(f"Row: {rgb_row}")
        cur_extrinsics = extrinsics[rgb_row]
        # R_W2C = cur_extrinsics[:, :3]
        # t_W2C = cur_extrinsics[:, 3]

        # convert_W2C_to_C2W
        converted_pose = np.dot(axis_transform, np.linalg.inv(cur_extrinsics))
        # print(converted_pose)
        cur_pose = np.hstack([converted_pose[:3], np.array([[H, W, focal]]).T]).astype(np.float32)

        if ind % testing_hold == 0:
            test_images.append(rgb)
            test_poses.append(cur_pose)
        else:
            train_images.append(rgb)
            train_poses.append(cur_pose)
            depth_info = {"depth": [], "coord": [], "error": []}
            for x in range(depth.shape[1]):
                for y in range(depth.shape[0]):
                    if depth[y, x] > 0:
                        depth_info["depth"].append(depth[y, x])
                        depth_info["coord"].append(np.array([x, y], dtype=np.float64))
                        depth_info["error"].append(1.0 + 2e-2)

            depth_info["depth"] = np.array(depth_info["depth"], dtype=np.float64)
            depth_info["coord"] = np.array(depth_info["coord"], dtype=np.float64)
            depth_info["error"] = np.array(depth_info["error"], dtype=np.float64)
            train_depths.append(depth_info)

    print("saving images.....")
    np.savez_compressed(join(save_dir, "train_images.npz"), np.array(train_images))
    np.savez_compressed(join(save_dir, "test_images.npz"), np.array(test_images))
    print("saving poses.....")
    np.save(join(save_dir, "train_poses.npy"), np.array(train_poses))
    np.save(join(save_dir, "test_poses.npy"), np.array(test_poses))
    print("saving depths.....")
    # np.save(join(save_dir, "train_depths.npy"), np.array(train_depths))
    np.savez_compressed(join(save_dir, "train_depths.npz"), np.array(train_depths))

    bounds = [min_bds, max_bds]
    print(f"Boundarys - Min:{min_bds}, Max: {max_bds}")
    print("saving bounds.....")
    np.save(join(save_dir, "bds.npy"), np.array(bounds))
    print("finished!")


if __name__ == "__main__":
    # base_dir = "../DSNeRF_data"
    base_dir = "../AnnaTrain"
    # convert(base_dir=base_dir, testing_hold=8, depth_start=0)

    # arr_depth = np.load(join(base_dir, "llff_colmap/train_depths.npz"), allow_pickle=True)
    # arr_depth = arr_depth[arr_depth.files[0]]
    # print(type(arr_depth[0]))
    # print(type(arr_depth[0]["depth"][0]))
    # print(arr_depth[0]["depth"])
    # print(type(arr_depth[0]["coord"][0, 0]))
    # print(arr_depth[0]["coord"])
    # print(type(arr_depth[0]["error"][0]))
    # print(arr_depth[0]["error"])

    # orchids_10view
    # arr_img = np.load(join(base_dir, "llff_colmap/train_images.npz"), allow_pickle=True)
    # arr_img = arr_img[arr_img.files[0]]
    # print(type(arr_img[0][0, 0, 0]))
    # print(arr_img[0].shape)
    # arr_pose = np.load(join(base_dir, "llff_colmap/test_poses.npy"), allow_pickle=True)
    # print(type(arr_pose[0, 0, 0]))
    # print(arr_pose[0])
