import os
import cv2
import sys
import math
import numpy as np
from PIL import Image
from plyfile import PlyData


# from scipy.spatial import ConvexHull


def sorted_nicely(l):
    """
    Sort the given iterable in the way that humans expect.
    :param l: a list
    :return:
    """
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def import_ply(file, scale=1):
    name, ext = os.path.splitext(file)
    vertexes_npy = name + '_vertexes.npy'
    faces_npy = name + '_faces.npy'

    if not os.path.isfile(vertexes_npy) or not os.path.isfile(faces_npy):
        ply = PlyData.read(file)

    if not os.path.isfile(vertexes_npy):
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']

        x *= scale
        y *= scale
        z *= scale

        res_vertexes = np.stack([x, y, z], axis=-1)
        np.save(vertexes_npy, res_vertexes)
    else:
        res_vertexes = np.load(vertexes_npy, allow_pickle=True)

    if not os.path.isfile(faces_npy):
        res_faces = []
        try:
            res_faces = ply['face'].data['vertex_indices']
            np.save(faces_npy, res_faces)
        except:
            pass
    else:
        res_faces = np.load(faces_npy, allow_pickle=True)

    return res_vertexes, res_faces


def project(pts_3d, pose, K):
    pts_2d = np.matmul(pts_3d, pose[:3, :3].T) + pose[:3, 3:].T
    pts_2d = np.matmul(pts_2d, K.T)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def purge_outliers(points_2d, max_x=640, max_y=480):
    # remove rows in points_2d where points_2d < 0
    # remove rows in points_2d where points_2d exeeds max_x or max_y

    copy = np.copy(points_2d[np.where((points_2d[:, 0] > -1) & (points_2d[:, 0] < max_x) &
                                      (points_2d[:, 1] > -1) & (points_2d[:, 1] < max_y))])

    return copy


def contains_outliers(vertixes, max_x=640, max_y=480):
    condition = np.any((vertixes[:, 0] < 0) | (vertixes[:, 0] > max_x - 1) |
                       (vertixes[:, 1] < 0) | (vertixes[:, 1] > max_y - 1))

    return condition


def spherical_flip(points, center, param):
    n = len(points)  # total n points
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    norm_points = np.linalg.norm(points, axis=1)  # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    r = np.repeat(max(norm_points) * np.power(10.0, param), n, axis=0)  # Radius of Sphere

    flipped_points_temp = 2 * np.multiply(np.repeat((r - norm_points).reshape(n, 1), len(points[0]), axis=1), points)
    # Apply Equation to get Flipped Points
    flipped_points = np.divide(flipped_points_temp, np.repeat(norm_points.reshape(n, 1), len(points[0]), axis=1))
    flipped_points += points

    return flipped_points


# def convex_hull(points):
#     points = np.append(points, [[0, 0, 0]], axis=0)  # All points plus origin
#     hull = ConvexHull(points)  # Visible points plus possible origin. Use its vertices property.

#     return hull


def contains_points(vertexes, points_2d):
    for vertex in vertexes:
        if not np.any(vertex == points_2d):
            return False

        return True


def load_K(json_file):
    import json

    with open(json_file) as fp:
        data = json.load(fp)
    K = np.eye(3)
    K[0, 0] = data['fx']
    K[1, 1] = data['fy']

    K[0, 2] = data['ppx']
    K[1, 2] = data['ppy']

    return K


def main(root_path):
    # Defining  used paths
    # desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    # desktop = os.path.join(os.path.join(os.environ['HOME']), 'Scrivania')
    # linemod_path = os.path.join(desktop, "LINEMOD")
    # real_path = os.path.join('./renders')

    # dir_path = os.path.join(real_path, "cat_movement")
    jpeg_dir = os.path.join(root_path, "JPEGImages")
    # rgb_dir = os.path.join(root_path, "rgb")
    poses_npy_path = os.path.join(root_path, "object_poses.npy")
    poses_dir_path = os.path.join(root_path, "poses")
    masks_path = os.path.join(root_path, "masks")
    # ply_path = os.path.join(dir_path, os.path.basename(dir_path) + ".ply")
    # ply_path = 'data/cat.ply'
    # ply_path = os.path.join(root_path, 'registeredScene.ply')
    ply_path = os.path.join(root_path, 'registered.ply')
    ply_scale = 1

    try:
        os.mkdir(masks_path)
    except OSError:
        print("Creation of the directory %s failed" % masks_path)
    else:
        print("Successfully created the directory %s " % masks_path)

    # Importing of the point cloud
    if os.path.isfile(ply_path):
        print("Importing ply...")
        ply, faces = import_ply(ply_path, scale=ply_scale)
        print("Points ply")
    else:
        print("Can't find %s" % ply_path)
        sys.exit(0)

    # ply = ply / 1000

    # Importing of the pose matrices
    print("Importing poses")
    if os.path.isfile(poses_npy_path):
        poses = np.load(poses_npy_path, allow_pickle=True)
    elif os.path.isdir(poses_dir_path):
        poses = []

        lst = os.listdir(poses_dir_path)
        lst = sorted_nicely(lst)

        for file in lst:
            if file.endswith(".txt"):  # Right now this if is totally useless but I still love him as a child
                pose = np.loadtxt(os.path.join(poses_dir_path, file), dtype='float', delimiter=' ')
                # pose = np.transpose(pose)
                poses.append(pose[:3][:])
    else:
        print("Can't find the poses")
        sys.exit(0)

    print("Poses imported")

    # Defining matrices of intrinsics parameters
    # K = np.array([[1.74785729e+03, 0.00000000e+00, 1.01313876e+03],
    #               [0.00000000e+00, 1.75132357e+03, 4.76604497e+02],
    #               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # kinect = np.array([[1081.37, 0., 959.5],
    #                    [0., 1081.37, 539.5],
    #                    [0., 0., 1.]])

    # linemod = np.array([[572.4114, 0., 325.2611],
    #                     [0., 573.57043, 242.04899],
    #                     [0., 0., 1.]])

    # kinect1 = np.array([[520., 0., 320.],
    #                     [0., 520., 240.],
    #                     [0., 0., 1.]])

    resolution = (480, 640)
    # K[0, 2], K[1, 2] = resolution[1] // 2, resolution[0] // 2

    current_K = load_K(os.path.join(root_path, 'intrinsics.json'))
    i = 0
    for p in poses:
        print("\n \n Analyzing frame ", i)
        i += 1

        # p = p[0:3, :]
        # p[2, 3] -= 0.17
        print(p)
        # Calculate 2d projection of the point cloud
        points_2d = project(ply, p, current_K)
        points_2d = points_2d.astype(np.int32)

        # Calculate the vertexes of the faces in the 2d plane
        print("Calculating 2d Vertexes")
        vertexes = []
        for indexes in faces:
            vertexes.append(points_2d[indexes])
        print("Vertexes calculated")

        # Convex Hull code block. -- Commented until further notice
        '''
        t = p[:, 3]
        c = np.array([t])
        flipped_cloud = spherical_flip(ply, c, math.pi)
        hull = convex_hull(flipped_cloud)
        visible_vertex = hull.vertices[:-1]  # indexes of visible points
        points_2d = points_2d[visible_vertex]
        '''

        # THE PURGE
        points_2d = purge_outliers(points_2d, resolution[1], resolution[0])

        # Create and color the image
        mask_image = np.zeros(resolution)
        mask_image[points_2d[:, 1], points_2d[:, 0]] = 255

        # Face coloration
        for v in vertexes:
            if contains_outliers(v, resolution[1], resolution[0]):
                continue
            v = np.asarray(v)  # asarray 3x2
            v = np.array([v])  # array 1x3x2
            mask_image = cv2.fillPoly(mask_image, v, 255)

        img = np.asarray(Image.open(jpeg_dir + '/{}.jpg'.format(str(i - 1))), dtype=np.uint8)

        # colorimg = np.asarray(Image.fromarray(img).resize((resolution[1] // 4, resolution[0] // 4)))
        # tmp = np.stack((mask_image, mask_image, mask_image), axis=-1).astype(np.uint8)
        # tmp = Image.fromarray(tmp)
        # mask_vis = np.asarray(tmp.resize((resolution[1] // 4, resolution[0] // 4)))
        # out = cv2.hconcat((cv2.cvtColor(colorimg, cv2.COLOR_RGB2BGR),
        #                   mask_vis.astype(np.uint8)))
        view_scale = 1
        x_scaled = int(resolution[0] * view_scale)
        y_scaled = int(resolution[1] * view_scale)

        colorimg = np.asarray(Image.fromarray(img).resize((y_scaled, x_scaled)))
        tmp = np.stack((mask_image, mask_image, mask_image), axis=-1).astype(np.uint8)
        tmp = Image.fromarray(tmp)
        mask_vis = np.asarray(tmp.resize((y_scaled, x_scaled)))
        out = cv2.hconcat((cv2.cvtColor(colorimg, cv2.COLOR_RGB2BGR),
                           mask_vis.astype(np.uint8)))

        # out = mask_image
        cv2.imshow('Frame+Mask', out)

        # mask out
        mask_image[mask_image > 0] = 1  # convert in binary
        cv2.imshow('Masked Out',
                   cv2.cvtColor(colorimg * np.stack((mask_image, mask_image, mask_image), axis=-1).astype(np.uint8),
                                cv2.COLOR_RGB2BGR))

        mask_path = os.path.join(masks_path, "{}.png".format(str(i - 1)))

        cv2.imwrite(mask_path, mask_image)

        if cv2.waitKey(33) == ord('q'):
            break


if __name__ == "__main__":
    # todo: for folder in glob.glob('LINEMOD/*')
    root_path = os.path.join(os.getcwd(), 'LINEMOD', 'new_acquisition')
    main(root_path)
    sys.exit(0)
