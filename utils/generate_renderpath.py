import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Slerp, Rotation

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    """
    """
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def generate_renderpath(poses, focal, N_views = 120, N_rots = 2, zrate=.5, scale=2):
    '''
    poses: N x 3 x 4
    '''
    # print(poses)
    # print("!!!!!!!!!!!!!!!!!!")
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    translation = poses[:,:3, 3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(translation), 90, axis=0) * scale

    render_poses = []
    rads = np.array(list(rads) + [1.])
    # zrate = 5
    for theta in np.linspace(0., 2.*np.pi*N_rots, N_views+1)[:-1]:
        # ind = 0
        # abstheta = theta % (2. * np.pi)
        # if abstheta < np.pi:
        #     ind = np.abs(abstheta - np.pi / 2)
        # else:
        #     ind = np.abs(abstheta - np.pi / 2 * 3)
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta),-np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def Slerp_path(poses, interpolate_node=1):
    """
    Parameters
    poses: [N, 3, 4] Lists of pose of rendering path
    interpolate_node: nodes number 

    Return
    render_poses: [m*(N-1)+N, 3, 4]
    """
    views_NUM = len(poses)
    inter_time = np.linspace(0., 1., num=interpolate_node+2)
    render_poses = [poses[0]]
    for v_idx in range(views_NUM-1):
        key_rots = Rotation.from_matrix(poses[v_idx:v_idx+2, :3, :3])
        slerp = Slerp([0.0, 1.0], key_rots)
        interp_rots = slerp(inter_time).as_matrix()
        # get the translation matrix of interpolate points
        trans_1 = poses[v_idx, :3, 3]
        trans_2 = poses[v_idx+1, :3, 3]
        for idx in range(interpolate_node):
            time_step = inter_time[idx+1]
            inter_trans = (1.0-time_step) * trans_1 + time_step * trans_2
            inter_pose = np.concatenate((interp_rots[idx+1], inter_trans[:,np.newaxis]), axis=1)
            render_poses.append(inter_pose)
        # add the end pose
        render_poses.append(poses[v_idx+1])
    print(f"Rendering poses number: {len(render_poses)}")
    return render_poses

