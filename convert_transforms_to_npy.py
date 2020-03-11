import os
import numpy as np
from tqdm import tqdm


def convert_transforms(root_path):
    file = os.path.join(root_path, 'transforms.npy')
    poses_path = os.path.join(root_path, 'poses')
    os.makedirs(poses_path, exist_ok=True)

    pose_file = os.path.join(poses_path, '{}.npy')
    poses_file = os.path.join(root_path, 'object_poses.npy')

    transforms = np.load(file)  # [n,4,4]
    n = transforms.shape[0]
    poses = np.zeros((n, 4, 4))  # [n,4,4]

    for i, curr_transform in enumerate(tqdm(transforms)):
        p = np.linalg.inv(curr_transform)  # calculate gt pose

        poses[i, :, :] = p
        np.save(pose_file.format(i), p)  # save current gt pose
        np.save(poses_file, poses)  # overwrite each step


if __name__ == "__main__":
    root = os.path.join(os.getcwd(), 'LINEMOD', 'new_acquisition')
    convert_transforms(root)
