import os
import open3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def register(source_pcd_file, target_pcd_file, threshold=0.02, init_trans=None):
    source = open3d.io.read_point_cloud(source_pcd_file)
    target = open3d.io.read_point_cloud(target_pcd_file)
    # threshold = 0.02
    if init_trans is None:
        init_trans = np.eye(4)
        # init_trans = np.asarray([[0.862, 0.011, -0.507, 0.5],
        #                          [-0.139, 0.967, -0.215, 0.7],
        #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, init_trans)
    print("Initial alignment")
    evaluation = open3d.registration.evaluate_registration(source, target,
                                                           threshold, init_trans)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = open3d.registration.registration_icp(
        source, target, threshold, init_trans,
        open3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = open3d.registration.registration_icp(
        source, target, threshold, init_trans,
        open3d.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)


if __name__ == "__main__":

    folder = os.path.join(os.getcwd(), 'LINEMOD', 'new_acquisition')
    target = os.path.join(folder, 'registeredScene.ply')
    source = os.path.join(folder, 'RadioWithBox_OriginBotRight.ply')

    if not os.path.isfile(target):
        print('pcd {} not found'.format(target))
        exit()

    if not os.path.isfile(source):
        print('pcd {} not found'.format(source))
        exit()

    init_trans = None
    # transforms = np.load(os.path.join(folder, 'transforms.npy'))
    # init_trans = np.linalg.inv(transforms[0, :, :])

    register(source, target, init_trans=init_trans)
