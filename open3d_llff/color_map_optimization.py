import open3d as o3d
from os.path import join, exists
from os import makedirs
import numpy as np
import cv2
from colmap_opt_utils import *
from sklearn.neighbors import NearestNeighbors

depth_nbrs = None
rgb_nbrs = None

def depth_to_colormapjet(depth):
    depth_color = depth.copy()
    min_d, max_d = np.min(depth_color), np.max(depth_color)
    depth_color = depth_color * 255. / (max_d - min_d) 
    depth_color = np.uint8(depth_color)
    depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)
    return depth_color


def load_depth_and_cam(dir_depth, poses, timings, timestamp, K_parameters_depth):
    global depth_nbrs
    if not depth_nbrs:
        depth_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(timings[:, 1].reshape(-1, 1))

    _, frame_number_depth = depth_nbrs.kneighbors(np.array(float(timestamp) + 0 * (10 ** 4)).reshape(-1, 1))
    frame_number_depth = frame_number_depth[0][0]

    filename_depth = join(dir_depth, '{:06d}.png'.format(frame_number_depth))
    print(f"loading depth image {filename_depth}")
    depth = load_depth(filename_depth)
    

    M_depth = poses[frame_number_depth, 1:].reshape(4, 4).copy()
    K_depth = K_parameters_depth[:9].reshape(3, 3) # intrinsics
    # M_depth[:3, 3] *= 1000 
    M_depth = np.dot(axis_transform, np.linalg.inv(M_depth))

    cam_depth = {}
    cam_depth['K_dist'] = K_depth  
    cam_depth['M_dist'] = M_depth 

    return depth, cam_depth

def load_rgb_and_cam(dir_rgb, poses_rgb, timing_rgb, time_stamp, K_parameters_rgb):
    global rgb_nbrs
    if not rgb_nbrs:
        rgb_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
            timing_rgb[:, 1].reshape(-1, 1))

    _, frame_number_rgb = rgb_nbrs.kneighbors(
        np.array(float(time_stamp) + 0 * (10 ** 4)).reshape(-1, 1))
    frame_number_rgb = frame_number_rgb[0][0]

    filename_rgb = join(dir_rgb, '{:06d}.png'.format(frame_number_rgb))
    print(f"loading rgb image {filename_rgb}")
    rgb = cv2.imread(filename_rgb)

    K_color = K_parameters_rgb[:9].reshape(3, 3)
    M_color = poses_rgb[frame_number_rgb, 1:].reshape(4, 4).copy() 
    cam_rgb = {}
    cam_rgb['M_origin'] = np.dot(axis_transform, np.linalg.inv(M_color))
    # M_color[:3, 3] *= 1000
    M_color = np.dot(axis_transform, np.linalg.inv(M_color))

    
    cam_rgb['K_color'] = K_color
    cam_rgb['M_color'] = M_color

    return rgb, cam_rgb, filename_rgb

def create_point_cloud_from_depth(depth, cam_depth, remove_outlier=True, remove_close_to_cam=300):
    '''
    output: point cloud in the depth frame
    '''
    K_depth = cam_depth['K_dist']

    img2d_converted = depthConversion(depth, K_depth[0][0], K_depth[0][2], K_depth[1][2]) # point depth to plane depth, basically, undistortion
    # img2d_converted_color = depth_to_colormapjet(img2d_converted) # plane depth color map jet
    # cv2.imshow('img2d_converted_color', img2d_converted_color)
    points = generatepointcloud(img2d_converted, K_depth[0][0], K_depth[1][1], K_depth[0][2], K_depth[1][2]) # in the depth coor
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if remove_outlier:
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=50)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=4.0)

    if remove_close_to_cam > 0:
        center = np.array([0, 0, 0])
        R = np.eye(3)
        extent = np.array([remove_close_to_cam, remove_close_to_cam, remove_close_to_cam])
        bb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        close_points_indices = bb.get_point_indices_within_bounding_box(pcd.points)
        pcd = pcd.select_by_index(close_points_indices, invert=True) #select outside points
    return pcd

def stitch_pcd(source, target, transformation):
    source.transform(transformation)
    # o3d.visualization.draw_geometries([source, target])
    return source + target


def generate_depth_maps(dir_seq, dir_depth, dir_rgb, dir_save, total_depth_frames=100, start_frame=0, depth_interval=1):

    if not exists(dir_save):
        makedirs(dir_save)
    ####### loading from dataset ######
    # Depth
    poses_depth = np.loadtxt(join(dir_depth, 'Pose.txt'))
    timing_depth = np.loadtxt(join(dir_depth, 'Timing.txt'))
    K_parameters_depth = np.loadtxt(join(dir_depth, 'Intrinsics.txt'))
    dist_coeffs = np.array(K_parameters_depth[9:14]).astype('float32')
    w_depth, h_depth = [int(x) for x in K_parameters_depth[-2:]]

    # RGB
    poses_rgb = np.loadtxt(join(dir_rgb, 'Pose.txt'))
    timing_rgb = np.loadtxt(join(dir_rgb, 'Timing.txt'))
    K_parameters_rgb = np.loadtxt(join(dir_rgb, 'Intrinsics.txt'))
    w_color, h_color = [int(x) for x in K_parameters_rgb[-2:]]

    ###### prepare for color map optimization ######
    # we use a continuous frames of videos as input for Color Map Optimization
    # initization of containers
    debug_mode = False
    start_frame = start_frame            # the first frame of video
    # total_depth_frames = 100    # total number of frames to be processed
    rgbd_images = []           # container for rgbd images
    camera_parameters = []     # container for camera intrinsic and extrinsic parameters
    whole_pcd = None           # Collection of while point clouds
    filenames_rgb = []         # rgb file paths
    for frame_number_depth in range(start_frame, start_frame+total_depth_frames*depth_interval, depth_interval):
        time_stamp = timing_depth[frame_number_depth, 1]
        # find the nearest depth frame
        depth, cam_depth_calib = load_depth_and_cam(
            dir_depth, poses_depth, timing_depth, time_stamp, K_parameters_depth
            )
        K_depth = cam_depth_calib['K_dist']
        depth_undistort = cv2.undistort(depth, K_depth, dist_coeffs, None, K_depth)
        if debug_mode:
            # visualize depth & undistorted depth
            cv2.imshow('depth_undistort', depth_to_colormapjet(depth_undistort))
            cv2.imshow('depth', depth_to_colormapjet(depth))
        # find the nearest rgb frame
        rgb, cam_rgb_calib, filename_rgb = load_rgb_and_cam(
            dir_rgb, poses_rgb, timing_rgb, time_stamp, K_parameters_rgb
            )
        # save rgb image path
        filenames_rgb.append(filename_rgb)

        # build and store point cloud
        # NOTE: pcd_colored in the world frame, pcd still in the depth frame
        pcd = create_point_cloud_from_depth(
            depth_undistort, cam_depth_calib,
            remove_outlier=True, remove_close_to_cam=1500
            ) # depth frame
        pcd_colored = get_colored_pcd(pcd, rgb, cam_rgb_calib, cam_depth_calib)
        # o3d.visualization.draw_geometries([pcd_colored])

        # Aligned Depth to be the same size with rgb image
        depth_aligned = map_depth_to_rgb(pcd, rgb, cam_rgb_calib, cam_depth_calib, reference='rgb')
        # cv2.imshow("depth_aligned", depth_to_colormapjet(depth_aligned))
        # o3d.visualization.draw_geometries([pcd_colored])
        # print(np.max(depth_aligned))
        np.save(join(dir_save, 'im_%d.npy' % frame_number_depth), depth_aligned)
    np.save(join(dir_save, 'rgb_paths.npy'), filenames_rgb)



if __name__ == "__main__":
    dir_seq = "../AnnaTrain" # keep the original file structure of this file 
    generate_depth_maps(dir_seq=dir_seq,
                        dir_depth=join(dir_seq, 'Depth'), # depth map dir
                        dir_rgb=join(dir_seq, 'Video'), # video pics dir 
                        dir_save=join(dir_seq, 'depth_colmap_opt'), # where to save the outputs
                        total_depth_frames = 150, # the total number of depth maps to process 
                        start_frame = 20, # the index of the first depth map 
                        depth_interval = 2)  # interval between each chosen depth map 
    