import numpy as np
import glob
from os.path import join, exists
from os import makedirs
import cv2


def convert(base_dir, testing_hold=5, depth_start=0):
    """testing_hold: one test image for every "testing_hold" images
        depth_start: not used, set this to zero (which depth map to start with)
    """
    bounds = [1, 10] # for now...
    train_images = []
    test_images = []
    train_poses = []
    test_poses = []
    train_depths = []

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
    focus = (int)(1e-5 * 1e-3 * (intrinsics[0] + intrinsics[4]))  # original unit: mm?
    H = cv2.imread(rgb_paths[0]).shape[0]
    W = cv2.imread(rgb_paths[0]).shape[1]

    # get all video extrinsics
    extrinsics = np.loadtxt(extrinsics_path)[:, 1:13]

    for ind in range(len(depths_paths)):
        # load depth and RGB
        rgb = cv2.imread(rgb_paths[ind])
        print(f"loaded rgb: {str(rgb_paths[ind])}")
        depth = 1e-3 * np.load(depths_paths[ind])# original unit: mm?
        print(f"loaded depth: {str(depths_paths[ind])}")

        # get current row of intrinsics and concat with HWF
        rgb_row = int(rgb_paths[ind].split("\\")[-1][:-4])
        cur_extrinsics = np.resize(np.array(extrinsics[rgb_row]), (3, 4))
        cur_pose = np.hstack([cur_extrinsics, np.array([[H], [W], [focus]])])

        if ind % testing_hold == 0:
            test_images.append(rgb)
            test_poses.append(cur_pose)
        else:
            train_images.append(rgb)
            train_poses.append(cur_pose)
            depth_info = {"depth": [], "coord": [], "error": []}
            for x in range(depth.shape[0]):
                for y in range(depth.shape[1]):
                    if depth[x, y] != 0:
                        depth_info["depth"].append(depth[x, y])
                        depth_info["coord"].append(np.array([x, y]))
                        depth_info["error"].append(1.0 + 2e-1)

            depth_info["depth"] = np.array(depth_info["depth"])
            depth_info["coord"] = np.array(depth_info["coord"])
            depth_info["error"] = np.array(depth_info["error"])
            train_depths.append(depth_info)

    print("saving images......\n")
    np.save(join(save_dir, "train_images.npy"), np.array(train_images))
    np.save(join(save_dir, "test_images.npy"), np.array(test_images))
    print("saving poses.....\n")
    np.save(join(save_dir, "train_poses.npy"), np.array(train_poses))
    np.save(join(save_dir, "test_poses.npy"), np.array(test_poses))
    np.save(join(save_dir, "video_poses.npy"), np.array(test_poses))
    print("saving depths.....\n")
    # np.save(join(save_dir, "train_depths.npy"), np.array(train_depths))
    np.savez_compressed(join(save_dir, "train_depths.npz"), np.array(train_depths))
    print("saving bounds.....\n")
    np.save(join(save_dir, "bds.npy"), np.array(bounds))
    print("finished!\n")


if __name__ == "__main__":
    base_dir = "../AnnaTrain"
    convert(base_dir=base_dir, testing_hold=8, depth_start=0)
    # arr = np.load(join(base_dir, "llff_colmap/train_depths.npy"), allow_pickle=True)
    # print(arr[0]["depth"])
    # print(arr[0]["coord"])
