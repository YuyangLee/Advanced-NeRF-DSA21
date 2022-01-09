import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    cam_to_world = trans_t(radius)
    cam_to_world = rot_phi(phi/180.*np.pi) @ cam_to_world
    cam_to_world = rot_theta(theta/180.*np.pi) @ cam_to_world
    cam_to_world = torch.Tensor(np.array(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ cam_to_world
    return cam_to_world


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    
    cam_dist = 512.
    scene_size = [512., 512., 512.]
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            if frame['file_path'][-4:] == '.png':
                fname = os.path.join(basedir, frame['file_path'])
            else:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
            if os.path.exists(fname):
                imgs.append(imageio.imread(fname))
            else:
                # magic code
                imgs.append(imageio.imread(
                    'data/nerf_synthetic/multihuman/render/000.png'))
            poses.append(np.array(frame['transform_matrix']))
            
            # Find the proper size for the 3D scene space
            cam_translation = np.asarray(frame['transform_matrix']).transpose()[-1, :3]
            norm = np.linalg.norm(cam_translation)
            if norm < cam_dist:
                cam_dist = norm
                scene_size = np.abs(cam_translation) * 0.95
            
        # keep all 4 channels (RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    if not meta.__contains__("camera_angle_x"):
        focal = meta["focal"]
    else:
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(
                img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
