
"""
record2.py
---------------

Main Function for recording a video sequence into cad (color-aligned-to-depth) 
images and depth images

Using librealsense SDK 2.0 with pyrealsense2 for SR300 and D series cameras


"""

# record for 40s after a 5s count down
# or exit the recording earlier by pressing q

import argparse
import sys
import os
import time
import cv2
import numpy as np
import logging
import json
import pyrealsense2 as rs
import png
RECORD_LENGTH = 0

logging.basicConfig(level=logging.INFO)
# from config.DataAcquisitionParameters import DEPTH_THRESH

parser = argparse.ArgumentParser(
    description='Record RGB and Depth using RealSense cameras')
parser.add_argument('--destination', type=str,
                    default=os.path.join('LINEMOD', 'new_acquistion'))
parser.add_argument('--rgb_folder', type=str, default='JPEGImages')
parser.add_argument('--depth_folder', type=str, default='depth')
parser.add_argument('--record_duration', type=int, default=60)
parser.add_argument('--warn_user', type=bool, default=False)

rgb_folder_name = None
depth_folder_name = None


def make_directories(folder):
    rgb_folder = os.path.join(folder, rgb_folder_name)
    depth_folder = os.path.join(folder, depth_folder_name)
    if not os.path.exists(rgb_folder):
        os.makedirs(rgb_folder)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)


def print_usage():

    print("Usage: record2.py <foldername>")
    print("foldername: path where the recorded data should be stored at")
    print("e.g., record2.py LINEMOD/mug")


def is_blurry(img, th=200):
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < th


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        # folder = sys.argv[1]+"/"
        folder = args.destination
        if (os.path.isdir(folder)):
            import uuid
            uuid = uuid.uuid4().hex
            folder = folder + '_{}'.format(uuid)
            print('Folder already exists, please delete or change name. Renaming current acquisition to: {}'.format(folder))

        rgb_folder_name = args.rgb_folder
        depth_folder_name = args.depth_folder

        RECORD_LENGTH = args.record_duration

        warn_user = args.warn_user
    except:
        print_usage()
        exit()

    FileName = 0
    make_directories(folder)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start pipeline
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Color Intrinsics
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }

    with open(os.path.join(folder, 'intrinsics.json'), 'w') as fp:
        json.dump(camera_parameters, fp)

    align_to = rs.stream.color
    align = rs.align(align_to)

    if warn_user:
        # warn user
        import winsound
        duration = 500  # milliseconds
        freq = 570  # Hz
        winsound.Beep(freq, duration)

    T_start = time.time()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())

        # Visualize count down

        if time.time() - T_start > 5:

            if is_blurry(c):
                continue

            # filecad= folder+"JPEGImages/%s.jpg" % FileName
            # filedepth= folder+"depth/%s.png" % FileName
            filecad = os.path.join(
                folder, rgb_folder_name, "{}.jpg".format(FileName))
            filedepth = os.path.join(
                folder, depth_folder_name, "{}.png".format(FileName))
            cv2.imwrite(filecad, c)
            with open(filedepth, 'wb') as f:
                writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = d.tolist()
                writer.write(f, zgray2list)

            FileName += 1
        if time.time() - T_start > RECORD_LENGTH + 5:
            pipeline.stop()
            break

        if time.time() - T_start < 5:
            cv2.putText(c, str(5-int(time.time() - T_start)), (240, 320),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0, 0, 255), 2, cv2.LINE_AA)
        if time.time() - T_start > RECORD_LENGTH:
            cv2.putText(c, str(RECORD_LENGTH+5-int(time.time()-T_start)), (240, 320),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('COLOR IMAGE', c)

        # press q to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.stop()
            break

    # Release everything if job is finished
    cv2.destroyAllWindows()

    if warn_user:
        # warn user
        import winsound
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
