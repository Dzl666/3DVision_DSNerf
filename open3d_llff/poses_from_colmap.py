import numpy as np
import glob
from os.path import join, exists
from os import makedirs
import cv2


def convert(base_dir, testing_hold=8):

    train_poses = []
    test_poses = []

    poses_path = join(base_dir, "poses_bounds.npy")
    save_dir = join(base_dir, "colmap")  # where to save the outputs
    if not exists(save_dir):
        makedirs(save_dir)

    poses_bds = np.load(poses_path)
    poses_arr = poses_bds[:, :-2].reshape([-1, 3, 5]) # N x 3 x 5
    bds = poses_bds[:, -2:].transpose([1,0])
    print(poses_arr.shape)

    for i in range(poses_arr.shape[0]):
        cur = poses_arr[i, :, :]
        cur[:, 4] = np.array([720., 1280., 988.]).T
        if i % testing_hold == 0:
            test_poses.append(cur.astype(np.float32))
        else:
            train_poses.append(cur.astype(np.float32))

    print("saving poses.....\n")
    np.save(join(save_dir, "train_poses_colmap.npy"), np.array(train_poses))
    np.save(join(save_dir, "test_poses_colmap.npy"), np.array(test_poses))

    bounds = [np.min(bds), np.max(bds)]
    print(f"Boundarys - Min:{np.min(bds)}, Max: {np.max(bds)}")
    print("saving bounds.....")
    np.save(join(save_dir, "bds_colmap.npy"), np.array(bounds))
    print("finished!")


if __name__ == "__main__":
    base_dir = "../AnnaTrain/66_20_3"
    convert(base_dir=base_dir, testing_hold=8)

