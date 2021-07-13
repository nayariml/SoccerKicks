from math import inf
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

smpl_joints = {
    1: 'mid_hip', 2: 'right_hip', 3: 'left_hip', 4: 'spine', 5: 'right_knee', 6: 'left_knee',
    7: 'chest', 8: 'right_ankle', 9: 'left_ankle', 10: 'thorax', 11: 'right_toe', 12: 'left_toe',
    13: 'neck', 14: 'right_thorax', 15: 'left_thorax', 16: 'head', 17: 'right_shoulder', 18: 'left_shoulder',
    19:'right_elbow', 20:'left_elbow', 21: 'right_wrist', 22:'left_wrist', 23: 'right_hand', 24: 'left_hand'
}


def joints_three(joints):#links - bones
    j = joints[0:] #take out the root

    return[[j[12], j[15]],[j[12], j[9]],[j[0], j[1]], [j[0],j[2]], [j[0],j[3]], [j[3], j[6]], [j[6], j[9]],[j[9],j[13]],[j[13],j[16]],
            [j[16],j[18]], [j[18], j[20]], [j[20], j[22]], [j[9], j[14]],[j[14], j[17]], [j[17], j[19]], [j[19], j[21]], [j[21], j[23]],
            [j[1], j[4]], [j[4], j[7]], [j[7], j[10]], [j[2], j[5]], [j[5], j[8]], [j[8],j[11]]]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def draw_skeleton(kintree, ouput_dir, ax=None):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=95, azim=-90)
    else:
        ax = ax

    colors = []
    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    for c in kintree_colors:
        colors += left_right_mid[c]
    # For the 25 joint locations
    for i in range(len(kintree)):

        j1 = kintree[i][0]
        j2 = kintree[i][1]
        ax.plot([j1[0], j2[0]],
                [j1[1]*-1, j2[1]*-1],
                [j1[2]*-1, j2[2]*-1],
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
    plt.savefig(ouput_dir)
    return

def make_image(joints, output_dir, frame):

  #mkdir(output_dir)
  kintree = joints_three(joints[frame])
  print('Draw skeleton')
  draw_skeleton(kintree, output_dir)
