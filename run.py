import os

# os.system('python record2.py LINEMOD/Radio')
os.system('python compute_gt_poses.py all')  # 'all' or 'LINEMOD/OBJNAME' [ REQ. TIME > 2h ]
os.system('python register_scene.py all')  # 'all' or 'LINEMOD/OBJNAME' [ REQ. TIME < 5 min ]
os.system('python register_segmented.py all')  # 'all' or 'LINEMOD/OBJNAME' [ REQ. TIME < 5 min ]
