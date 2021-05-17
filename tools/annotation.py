#Implemented by @nayariml 
import os
import csv
import cv2
import json
import pickle
import datetime
import os.path as osp
from time import time

from evaluate import compute_errors_function
from read2DFiles import digest_sys


# Predictions.
#'cams', 'joints', 'kps', 'poses', shapes', 'verts', 'omegas'
joints_names_dic = {}

# The names of universal 25 joints with toes.
joints_names_dic={

    0: 'Right_heel',
    1: 'Right_knee',
    2: 'Right_hip',
    3: 'Left_hip',
    4: 'Left_knee',
    5: 'Left_heel',
    6: 'Right_wrist',
    7: 'Right_elbow',
    8: 'Right_shoulder',
    9: 'Left_shoulder',
    10: 'Left_elbow',
    11: 'Left_wrist',
    12: 'Neck',
    13: 'Head_top',
    14: 'nose',
    15: 'left_eye',
    16: 'right_eye',
    17: 'left_ear',
    18: 'right_ear',
    19: 'left_big toe',
    20: 'right_big toe',
    21: 'Left_small toe',
    22: 'Right_small toe',
    23: 'Left_ankle',
    24: 'Right_ankle'
}

def check_points(all_kps, joint):

    error2d = {}; error3d = {}; count_kps = 0; count_joints = 0; kps = []; joints = []

    for i in range(25):

        if all_kps[i][0] == 0 or all_kps[i][1] == 0:
            count_kps += 1
            kps = [joints_names_dic[i], all_kps[i][0], all_kps[i][1]]

        if joint[i][0] == 0 or joint[i][1] == 0 or joint[i][2] == 0:
            count_joints += 1
            joints = [joints_names_dic[i], joint[i][0], joint[i][1], joint[i][2]]

    error2d = {'count_error': count_kps, 'error2d': kps}
    error3d = {'count_error': count_joints, 'error3d': joints}
    return error2d, error3d


def get_kps_labels(all_kps, joints):

    dic = {}

    dic={
        joints_names_dic[0]: {'2D_kps' : {'x': all_kps[0][0], 'y' : all_kps[0][1]}, '3D_joints' : {'x': joints[0][0], 'y' : joints[0][1], 'z': joints[0][2]}},
        joints_names_dic[1]: {'2D_kps' : {'x': all_kps[1][0], 'y' : all_kps[1][1]}, '3D_joints' : {'x': joints[1][0], 'y' : joints[1][1], 'z': joints[1][2]}},
        joints_names_dic[2]: {'2D_kps' : {'x': all_kps[2][0], 'y' : all_kps[2][1]}, '3D_joints' : {'x': joints[2][0], 'y' : joints[2][1], 'z': joints[2][2]}},
        joints_names_dic[3]: {'2D_kps' : {'x': all_kps[3][0], 'y' : all_kps[3][1]}, '3D_joints' : {'x': joints[3][0], 'y' : joints[3][1], 'z': joints[3][2]}},
        joints_names_dic[4]: {'2D_kps' : {'x': all_kps[4][0], 'y' : all_kps[4][1]}, '3D_joints' : {'x': joints[4][0], 'y' : joints[4][1], 'z': joints[4][2]}},
        joints_names_dic[5]: {'2D_kps' : {'x': all_kps[5][0], 'y' : all_kps[5][1]}, '3D_joints' : {'x': joints[5][0], 'y' : joints[5][1], 'z': joints[5][2]}},
        joints_names_dic[6]: {'2D_kps' : {'x': all_kps[6][0], 'y' : all_kps[6][1]}, '3D_joints' : {'x': joints[6][0], 'y' : joints[6][1], 'z': joints[6][2]}},
        joints_names_dic[7]: {'2D_kps' : {'x': all_kps[7][0], 'y' : all_kps[7][1]}, '3D_joints' : {'x': joints[7][0], 'y' : joints[7][1], 'z': joints[7][2]}},
        joints_names_dic[8]: {'2D_kps' : {'x': all_kps[8][0], 'y' : all_kps[8][1]}, '3D_joints' : {'x': joints[8][0], 'y' : joints[8][1], 'z': joints[8][2]}},
        joints_names_dic[9]: {'2D_kps' : {'x': all_kps[9][0], 'y' : all_kps[9][1]}, '3D_joints' : {'x': joints[9][0], 'y' : joints[9][1], 'z': joints[9][2]}},
        joints_names_dic[10]: {'2D_kps' : {'x': all_kps[10][0], 'y' : all_kps[10][1]}, '3D_joints' : {'x': joints[10][0], 'y' : joints[10][1], 'z': joints[10][2]}},
        joints_names_dic[11]: {'2D_kps' : {'x': all_kps[11][0], 'y' : all_kps[11][1]}, '3D_joints' : {'x': joints[11][0], 'y' : joints[11][1], 'z': joints[11][2]}},
        joints_names_dic[12]: {'2D_kps' : {'x': all_kps[12][0], 'y' : all_kps[12][1]}, '3D_joints' : {'x': joints[12][0], 'y' : joints[12][1], 'z': joints[12][2]}},
        joints_names_dic[13]: {'2D_kps' : {'x': all_kps[13][0], 'y' : all_kps[13][1]}, '3D_joints' : {'x': joints[13][0], 'y' : joints[13][1], 'z': joints[13][2]}},
        joints_names_dic[14]: {'2D_kps' : {'x': all_kps[14][0], 'y' : all_kps[14][1]}, '3D_joints' : {'x': joints[14][0], 'y' : joints[14][1], 'z': joints[14][2]}},
        joints_names_dic[15]: {'2D_kps' : {'x': all_kps[15][0], 'y' : all_kps[15][1]}, '3D_joints' : {'x': joints[15][0], 'y' : joints[15][1], 'z': joints[15][2]}},
        joints_names_dic[16]: {'2D_kps' : {'x': all_kps[16][0], 'y' : all_kps[16][1]}, '3D_joints' : {'x': joints[16][0], 'y' : joints[16][1], 'z': joints[16][2]}},
        joints_names_dic[17]: {'2D_kps' : {'x': all_kps[17][0], 'y' : all_kps[17][1]}, '3D_joints' : {'x': joints[17][0], 'y' : joints[17][1], 'z': joints[17][2]}},
        joints_names_dic[18]: {'2D_kps' : {'x': all_kps[18][0], 'y' : all_kps[18][1]}, '3D_joints' : {'x': joints[18][0], 'y' : joints[18][1], 'z': joints[18][2]}},
        joints_names_dic[19]: {'2D_kps' : {'x': all_kps[19][0], 'y' : all_kps[19][1]}, '3D_joints' : {'x': joints[19][0], 'y' : joints[19][1], 'z': joints[19][2]}},
        joints_names_dic[20]: {'2D_kps' : {'x': all_kps[20][0], 'y' : all_kps[20][1]}, '3D_joints' : {'x': joints[20][0], 'y' : joints[20][1], 'z': joints[20][2]}},
        joints_names_dic[21]: {'2D_kps' : {'x': all_kps[21][0], 'y' : all_kps[21][1]}, '3D_joints' : {'x': joints[21][0], 'y' : joints[21][1], 'z': joints[21][2]}},
        joints_names_dic[22]: {'2D_kps' : {'x': all_kps[22][0], 'y' : all_kps[22][1]}, '3D_joints' : {'x': joints[22][0], 'y' : joints[22][1], 'z': joints[22][2]}},
        joints_names_dic[23]: {'2D_kps' : {'x': all_kps[23][0], 'y' : all_kps[23][1]}, '3D_joints' : {'x': joints[23][0], 'y' : joints[23][1], 'z': joints[23][2]}},
        joints_names_dic[24]: {'2D_kps' : {'x': all_kps[24][0], 'y' : all_kps[24][1]}, '3D_joints' : {'x': joints[24][0], 'y' : joints[24][1], 'z': joints[24][2]}},
        }

    error2d, error3d = check_points(all_kps, joints)

    return dic, error2d, error3d

def mkdir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

def video_opencv(video_path):

    video = cv2.VideoCapture(video_path)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_time = str(datetime.timedelta(seconds=int(frame_count / fps)))

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    return video_time, frame_count, width, height

def main(preds_alpha, preds_open, track_id, save_dir, ouput_dir, name_file, video_path, main_dir):

    #ouput_dir = save_dir + '/annotations/'

    mkdir(ouput_dir)

    action = name_file[3:-1]

    video_length, video_frame_count, width, height = video_opencv(video_path)

    if preds_open:

        dic = {}; dic_renderer = {}; error2d_out = []; error3d_out = []

        ouput_path = ouput_dir + '/openpose_hmmr_annotations/'

        mkdir(ouput_path)

        all_kps_op = preds_open['kps'] #2D kps
        joints_op = preds_open ['joints'] #3D 25 joints
        poses_op = preds_open ['poses'] #SMPL rotation
        cams_op = preds_open['cams']
        verts_op = preds_open['verts']
        shapes_op = preds_open ['shapes']

        frame_count = joints_op.shape[0] -1
        total_frame_open = joints_op.shape[0]
        print("There are totally {} frames-OpenPose-HMRR".format(joints_op.shape[0]))

        for i in range(0, joints_op.shape[0]):

            nf = frame_count - i
            frame_index = 'frame_'+"%04d" %i + '/' + str(nf)
            frame_index_path = 'frame_'+"%04d" %i

            dic_points, error2d, error3d = get_kps_labels(all_kps_op[i], joints_op[i])

            dic = {
                'frame': frame_index,
                'action': action,
                'joints': dic_points,
            }

            with open(ouput_path + frame_index_path + '_joints.json', 'w') as jf:
                json.dump(str(dic), jf, ensure_ascii=False)

            if error2d['count_error'] != 0 or error3d['count_error'] != 0:
                print('There are unestimated joints - OpenPose - HMMR output')

                error2d_out.append([frame_index, error2d])
                error3d_out.append([frame_index,error3d])

                with open(ouput_dir + 'error_openpose_hmmr.txt', 'w') as f:
                    f.write(str(error2d_out) + "\n")
                    f.write(str(error3d_out) + "\n")
        openpose = True
        error2d_open = error2d['count_error']
        error3d_open = error3d['count_error']

    if preds_alpha:

        dic = {}; dic_renderer = {}; error2d_out = []; error3d_out = []

        ouput_path = ouput_dir + '/alphapose_hmmr_annotations/'

        mkdir(ouput_path)

        all_kps = preds_alpha['kps'] #2D kps
        joints = preds_alpha ['joints'] #3D 25 joints
        poses = preds_alpha ['poses'] #SMPL rotation
        cams = preds_alpha['cams']
        verts = preds_alpha['verts']
        shapes = preds_alpha ['shapes']

        frame_count = joints.shape[0] -1
        total_frame_alpha = joints.shape[0]
        print("There are totally {} frames-AlphaPose-HMRR".format(joints.shape[0]))

        for i in range(0, joints.shape[0]):

            nf = frame_count - i
            frame_index = 'frame_'+"%04d" %i + '/' + str(nf)
            frame_index_path = 'frame_'+"%04d" %i

            dic_points, error2d, error3d = get_kps_labels(all_kps[i], joints[i])

            dic = {
                'frame': frame_index,
                'action': action,
                'track_id': track_id,
                'joints': dic_points}

            with open(ouput_path + frame_index_path + '_joints.json', 'w') as jf:
                json.dump(str(dic), jf, sort_keys=True, indent=4)

            if error2d['count_error'] != 0 or error3d['count_error'] != 0:
                print('There are unestimated joints - AlphaPose-HMMR output')

                error2d_out.append([frame_index, error2d])
                error3d_out.append([frame_index,error3d])

                with open(ouput_dir + 'error_alphapose_hmmr.txt', 'w') as f:
                    f.write(str(error2d_out) + "\n")
                    f.write(str(error3d_out) + "\n")

        error2d_alpha = error2d['count_error']
        error3d_alpha = error3d['count_error']

        if not preds_open:
            openpose = False

        if error2d_alpha == 0 and error3d_alpha ==0 and openpose==True:
            print('Evaluating the metrics')

            str_error, errors_dict = compute_errors_function(
                preds_open['joints'],
                preds_alpha['joints'],
                #preds_open['poses'],
                #preds_alpha['poses']
                )
            print('\n')
            print('AlphaPose is the groud truth:', '\n')
            print(str_error)
            """
            with open(ouput_dir + 'evaluation.pkl', 'wb') as f:
                print('Saving evaluation results to', ouput_dir)
                pickle.dump(errors_dict, f)
            """

        if not preds_open:
            errors_dict = {}
            total_frame_open = None
            errors_dict = {
                'MPJPE': None,
                'MPJPE_PA': None,
                'PCK': None,
                'AUC': None
            }
            total_frame_open = None
            error2d_open = None
            error3d_open = None

        save_csv = video_path[:-31] + '3D_joints_info.csv'

        print('Saving evaluation results to', save_csv)

        save = [[name_file, int(video_frame_count), video_length, [width, height], total_frame_alpha, total_frame_open,error2d_alpha, error3d_alpha, error2d_open, error3d_open, errors_dict['PCK'], errors_dict['AUC']]]

        with open(save_csv, "a") as csvfile:
            file_is_empty = os.stat(save_csv).st_size == 0
            headers = ['id_Video_action','Video_input_frames','Video_input_lenght', 'Video_input_Resolution [width, height]', 'Hmmr-alphapose_output_frames','Hmmr-openpose_output_frames', 'error2d_alpha', 'error3d_alpha', 'error2d_open', 'error3d_open', 'PCK', 'AUC']
            writer = csv.writer(csvfile)
            if file_is_empty:
                writer.writerow(headers)
            writer.writerows(save)

    if not preds_alpha:

        errors_dict = {}
        total_frame_open = None
        errors_dict = {
            'MPJPE': None,
            'MPJPE_PA': None,
            'PCK': None,
            'AUC': None
        }
        total_frame_alpha = None
        error2d_alpha = None
        error3d_alpha = None

        total_frame_open = joints_op.shape[0]

        save_csv = main_dir + '/3D_joints_info.csv'

        print('Saving evaluation results to', save_csv)

        save = [[name_file, int(video_frame_count), video_length, [width, height], total_frame_alpha, total_frame_open, error2d_alpha, error3d_alpha, error2d_open, error3d_open, errors_dict['PCK'], errors_dict['AUC']]]

        with open(save_csv, "a") as csvfile:
            file_is_empty = os.stat(save_csv).st_size == 0
            headers = ['id_Video_action','Video_input_frames','Video_input_lenght', 'Video_input_Resolution [width, height]', 'Hmmr-alphapose_output_frames','Hmmr-openpose_output_frames', 'error2d_alpha', 'error3d_alpha', 'error2d_open', 'error3d_open', 'PCK', 'AUC']
            writer = csv.writer(csvfile)
            if file_is_empty:
                writer.writerow(headers)
            writer.writerows(save)

    print('Step 1 Done!')

    return

def read_file(alphapose_in, openpose_in):

    if not osp.exists(alphapose_in):
        preds_alpha = False
        print('AlphaPose-Hmmr rendered is not available')
    else:
        with open(alphapose_in, 'rb') as f:
            preds_alpha = pickle.load(f)
        f.close()
        print('AlphaPose-Hmmr rendered is available')

    if not osp.exists(openpose_in):
        preds_open = False
        print('OpenPose-Hmmr rendered is not available')
    else:
        with open(openpose_in, 'rb') as f:
            preds_open = pickle.load(f)
        f.close()
        print('OpenPose-Hmmr rendered is available')

    return preds_alpha, preds_open

if __name__ == '__main__':

    main_dir = '/home/nayari/projects/SoccerKicks//'

    videos_dir = main_dir + 'Original_videos/'

    dir = main_dir + '/Rendered/'

    action = 'Freekick/' #Penalty

    name_file = '22_freekick' #

    video_path = videos_dir + name_file + '.mp4'

    #save_dir = dir + action + name_file
    input_dir = dir + action + name_file

    ouput_dir = dir + action + name_file + '/annotations/'

    preds_file = 'hmmr_output.pkl'

    input_alpha =  dir + action + name_file + '/hmmr_output/'
    alphapose_in = input_alpha + preds_file
    openpose_in = dir + action + name_file + '/hmmr_output_openpose/'+ preds_file

    #track_id = alphapose_in[:1]
    track_id = 0

    ############################################################################
    print('\n')
    print('Starting 2D Keypoints annotations', '\n')

    digest_sys(track_id, input_dir, input_alpha, ouput_dir, main_dir, name_file)

    print('Starting 3D prediction annotations', '\n')

    preds_alpha, preds_open = read_file(alphapose_in, openpose_in)

    main(preds_alpha, preds_open, track_id, input_dir, ouput_dir, name_file, video_path, main_dir)

    print('Step 2 Done!')
