from math import inf
import os
import subprocess
import pickle
import ipdb
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

HEAD = 13
NECK = 12

LEFT_SHOULDER = 9
RIGHT_SHOULDER = 8
LEFT_ELBOW = 10
RIGHT_ELBOW = 7
LEFT_WRIST = 11
RIGHT_WRIST = 6

LEFT_HIP = 3
RIGHT_HIP = 2

LEFT_KNEE = 4
RIGHT_KNEE = 1

LEFT_HEEL = 5
RIGHT_HEEL = 0
LEFT_BTOE = 19
RIGHT_BTOE = 20

LEFT_STOE = 21
RIGHT_STOE = 22

LEFT_ANKLE = 23
RIGHT_ANKLE = 24

LEFT_EAR = 17
RIGHT_EAR = 18

NOSE = 14
LEFT_EYE = 15
RIGHT_EYE = 16

joints= {
    0: 'Right_heel', 1: 'Right_knee', 2: 'Right_hip', 3: 'Left_hip', 4: 'Left_knee',
    5: 'Left_heel', 6: 'Right_wrist',7: 'Right_elbow', 8: 'Right_shoulder', 9: 'Left_shoulder',
    10: 'Left_elbow', 11: 'Left_wrist', 12: 'Neck', 13: 'Head_top', 14: 'Nose', 15: 'Left_eye',
    16: 'Right_eye', 17: 'Left_ear', 18: 'Right_ear', 19: 'Left_big_toe', 20: 'Right_big_toe',
    21: 'Left_small_toe', 22: 'Right_small_toe', 23: 'Left_ankle', 24: 'Right_ankle'
}

def torso(j): #mid hip
    hx = (j[2][0] + j[3][0])/2
    hy = (j[2][1] + j[3][1])/2
    hz = (j[2][2] + j[3][2])/2
    return [hx, hy,hz]

def joints_three(joints):#links - bones
    j = joints
    MID_HIP = torso(j)

    return[[j[RIGHT_EYE], j[LEFT_EYE]], [j[LEFT_EAR],j[RIGHT_EAR]], [j[RIGHT_EYE],j[RIGHT_EAR]], [j[LEFT_EAR],j[LEFT_EYE]], [j[RIGHT_EYE], j[NOSE]], [j[LEFT_EYE], j[NOSE]],[j[HEAD],j[NECK]],[j[NECK],j[LEFT_SHOULDER]],
            [j[NECK],j[RIGHT_SHOULDER]], [j[RIGHT_SHOULDER], j[RIGHT_ELBOW]], [j[RIGHT_SHOULDER], j[RIGHT_ELBOW]],
            [j[LEFT_SHOULDER], j[LEFT_ELBOW]],[j[RIGHT_ELBOW], j[RIGHT_WRIST]], [j[LEFT_ELBOW], j[LEFT_WRIST]], [j[NECK], MID_HIP], [MID_HIP, j[RIGHT_HIP]],
            [j[RIGHT_HIP], j[RIGHT_KNEE]], [j[RIGHT_KNEE], j[RIGHT_ANKLE]], [j[RIGHT_ANKLE], j[RIGHT_HEEL]], [j[RIGHT_HEEL], j[RIGHT_BTOE]], [j[RIGHT_BTOE],
            j[RIGHT_STOE]], [j[RIGHT_STOE],j[RIGHT_HEEL]], [MID_HIP, j[LEFT_HIP]], [j[LEFT_HIP], j[LEFT_KNEE]], [j[LEFT_KNEE], j[LEFT_ANKLE]], [j[LEFT_ANKLE],
            j[LEFT_HEEL]], [j[LEFT_HEEL], j[LEFT_BTOE]], [j[LEFT_BTOE], j[LEFT_STOE]],[j[LEFT_STOE],j[LEFT_HEEL]]]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return

def draw_skeleton_frames(fi, _3Djoints, kintree, save_dir, ax=None):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=95, azim=270)
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
                [j1[2], j2[2]], #[j1[2]*-1, j2[2]*-1]
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)

    ouput_dir = save_dir + '/frame{:04d}.png'.format(fi)
    print(fi)
    plt.savefig(ouput_dir)
    return ax

def make_video(output_path, img_dir, fps=25):
        """
        output_path is the final mp4 name
        img_dir is where the images to make into video are saved.
        """
        cmd = [
            'ffmpeg',
            '-y',
            '-threads', '16',
            '-framerate', str(fps),
            '-i', '{img_dir}/frame%04d.png'.format(img_dir=img_dir),
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-an',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            output_path,
        ]
        print(' '.join(cmd))
        try:
            err = subprocess.call(cmd)
            if err:
                ipdb.set_trace()
        except OSError:
            ipdb.set_trace()
            print('OSError')


def make_image(preds_path, save_dir, file_name, frame):

    mkdir(save_dir)

    with open(preds_path, 'rb') as f:
        preds = pickle.load(f)
    f.close()

    print('Get joints prediction')

    _3Djoints = preds['joints']

    print('Len of video', len(_3Djoints))

    len_video = len(_3Djoints)

    for i in range(len_video):

        kintree = joints_three(_3Djoints[i])
        print('Draw skeleton, frame', i)
        draw_skeleton_frames(i, _3Djoints[i], kintree, save_dir)

    save = save_dir + file_name + '.mp4'

    make_video(save, save_dir)


if __name__ == '__main__':

    #Should have the 3D joint coordinates [x,y,z] for each joint in the kinematic skeleton

    main_dir = '/home/nayari/projects/SoccerKicks'

    dir = main_dir + '/Rendered/'

    action = 'Penalty/' #Penalty Freekick

    file_name = '14_penalty' #

    input_dir = dir + action + file_name

    ouput_dir = main_dir + '/animations/' + file_name + '/'

    preds_file = 'hmmr_output.pkl'

    input_alpha =  dir + action + file_name + '/hmmr_output/'

    alphapose_in = input_alpha + preds_file

    openpose_in = dir + action + file_name + '/hmmr_output_openpose/'+ preds_file

    make_image(openpose_in, ouput_dir, file_name, frame = 16)
