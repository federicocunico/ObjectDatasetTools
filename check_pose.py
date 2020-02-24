import open3d
import os
from tqdm import tqdm
import glob
import numpy as np
import cv2
from plyfile import PlyData
import json
import argparse
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(
    description='Record RGB and Depth using RealSense cameras')
parser.add_argument('--destination', type=str,
                    default="all")


def load_ply(model_path, subsample=1):
    npy = model_path[:-4]
    npy = npy + '.npy'
    if os.path.isfile(npy):
        pts = np.load(npy)
    else:
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']

        pts = np.stack([x, y, z], axis=-1)
        np.save(npy, pts)

    if subsample > 1:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        x = x[::subsample]
        y = y[::subsample]
        z = z[::subsample]
        pts = np.stack([x, y, z], axis=-1)
    return pts


def sorted_nicely(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def print_usage():
    print("Usage: record2.py --destination <folder>")
    print("foldername: path where the recorded data should be")
    print("e.g., record2.py --destination LINEMOD/mug")


def project_points(pts3d, RT, K, opencv=False):
    if opencv:
        tvec = RT[0:3, 3]
        R = RT[0:3, 0:3]
        rvec = cv2.Rodrigues(R)[0]
        pts2d, _ = cv2.projectPoints(pts3d, tvec, rvec, K, None)
        pts2d = pts2d.squeeze()
    else:
        # raise NotImplementedError(
        #     "Projection without opencv currently not implemented")
        RT = RT[0:3, :]
        pts2d = np.matmul(pts3d, RT[:, :3].T) + RT[:, 3:].T
        pts2d = np.matmul(pts2d, K.T)
        pts2d = pts2d[:, :2] / pts2d[:, 2:]

    return pts2d


def draw_model(rgb, pts3d, pose, K):
    pts2d = project_points(pts3d, pose, K)
    for p in pts2d:
        if len(p.shape) > 1:
            center = int(round(p[0, 0])), int(round(p[0, 1]))
        else:
            center = int(round(p[0])), int(round(p[1]))
        rgb = cv2.circle(rgb, center=center, radius=1,
                         color=(255, 0, 0), thickness=1)
    return rgb


def view3d(pts3d, vis=None):
    if pts3d.shape == (3,):
        pts3d = pts3d.reshape((1, 3))

    assert (len(pts3d.shape) == 2 and pts3d.shape[1] == 3)  # must be [n,3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts3d)
    if vis is None:
        open3d.visualization.draw_geometries([pcd])
    else:
        vis.add_geometry(pcd)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        # folder = sys.argv[1]+"/"
        dest = args.destination
        if dest == "all":
            folders = glob.glob("LINEMOD/*/")
        else:
            if (not os.path.isdir(dest)):
                print('Folder does not exists: {}'.format(dest))
                exit()
            folders = [dest]
    except:
        print_usage()
        exit()

    for folder in folders:
        print('Processing: {}'.format(folder))
        poses = np.load(os.path.join(folder, 'transforms.npy'))
        with open(os.path.join(folder, 'intrinsics.json'), 'r') as f:
            camera_intrinsics = json.load(f)

        rgbs = sorted_nicely(
            glob.glob(os.path.join(folder, 'JPEGImages', "*")))

        K = np.eye(3)
        K[0, 0] = camera_intrinsics['fx']
        K[1, 1] = camera_intrinsics['fy']
        K[0, 2] = camera_intrinsics['ppx']
        K[1, 2] = camera_intrinsics['ppy']
        print(K)

        model_path = os.path.join(folder, 'registeredScene.ply')
        obj = load_ply(model_path, subsample=200)

        # 3d visualization:
        trasl = poses[:, 0:3, 3]  # [n,3]
        n = trasl.shape[0]
        translations = np.zeros((n + 1, 3))  # add zero/ origin
        translations[1:, :] = trasl
        # view3d(translations)

        # 3d visualization:
        vis = open3d.visualization.Visualizer()
        vis.create_window()

        # projection
        for i, p in enumerate(tqdm(poses)):
            print(p)
            rgb = cv2.imread(rgbs[i])
            rgb = draw_model(rgb, obj, p, K)
            cv2.imshow('rgb', rgb)
            view3d(translations[0:i, :], vis)
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
        vis.destroy_window()
