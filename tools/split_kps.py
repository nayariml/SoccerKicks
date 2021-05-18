import os
import csv
import math
import json
import ipdb
import numpy as np
from glob import glob
import os.path as osp
import subprocess

def splitFileAlpha(out_json, nFrame, track_id):
    all_kps = read_person(out_json, nFrame)
    #all_kps = read_poseflow(out_json, nFrame)

    # Here we set which track to use.
    track_id_ = min(track_id, len(all_kps) - 1)
    print('Total number of PoseFlow tracks:', len(all_kps))
    print('Processing track_id:', track_id)
    kps = all_kps[track_id_]
    bboxes = []
     #End verification
    for i in range(len(kps)):
        if kps[i] is not None:
            bboxes.append(kps[i])
    #print(len(bboxes))
    if len(bboxes) != nFrame:
        all_kps = read_poseflow(out_json, nFrame)
        print('Not all frames have people detected in it.', '\n')
        # Here we set which track to use.
        track_id_ = min(track_id, len(all_kps) - 1)
        print('Total number of PoseFlow tracks:', len(all_kps))
        print('Processing track_id:', track_id)
        kps = all_kps[track_id_]
        bboxes = []
         #End verification
        for i in range(len(kps)):
            if kps[i] is not None:
                bboxes.append(kps[i])
        if len(bboxes) != nFrame:
            print(len(bboxes))
            ipdb.set_trace()
    return bboxes
def read_person(json_path, nFrame):

    min_kp_count=20
    with open(json_path, 'r') as f:
        data = json.load(f)
    if len(data.keys()) < nFrame:
       print('Not all frames have people detected in it.', '\n')
       ipdb.set_trace()
    print('AlphaPose json files: ok', '\n')

    kps = []
    for i in range(len(data)):
        key = str(i) + '.jpg'
        for people in data[key]:
            kp = np.array(people['keypoints']).reshape(-1, 3)
            kps.append(kp)

    #print(kps)
    #print(len(kps[0]))
    return [kps]

def read_poseflow(json_path, nFrame): #AlphaPose
    min_kp_count=20
    with open(json_path, 'r') as f:
        data = json.load(f)
    if len(data.keys()) < nFrame:
       print('Not all frames have people detected in it.', '\n')
       ipdb.set_trace()
    print('AlphaPose json files: ok', '\n')
    kps_dict = {}
    kps_count = {}
    for i, key in enumerate(data.keys()):
        # People who are visible in this frame.
        track_ids = []

        for person in data[key]:
            kps = np.array(person['keypoints']).reshape(-1, 3)
            idx = int(person['idx'])
            if idx not in kps_dict.keys():
                # If this is the first time, fill up until now with None
                kps_dict[idx] = [None] * i
                kps_count[idx] = 0
            # Save these kps.
            kps_dict[idx].append(kps)
            track_ids.append(idx)
            kps_count[idx] += 1
        #If any person seen in the past is missing in this frame, add None.
        for idx in set(kps_dict.keys()).difference(track_ids):
            kps_dict[idx].append(None)

    kps_list = []
    counts_list = []
    for k in kps_dict:
        if kps_count[k] >= min_kp_count:
            kps_list.append(kps_dict[k])
            counts_list.append(kps_count[k])

    #Sort it by the length so longest is first:
    sort_idx = np.argsort(counts_list)[::-1]
    kps_list_sorted = []
    for sort_id in sort_idx:
        kps_list_sorted.append(kps_list[sort_id])

    return kps_list_sorted

def keyAlpha(dirOut2, alphajson, nFrame, track_id):
    nFrame = nFrame

    out_json = osp.join(dirOut2, 'alphapose-results-forvis-tracked.json')
    if osp.exists(out_json):
        print('Tracking: done!')
        newKps = splitFileAlpha(out_json, nFrame, track_id)
    else:
        print('Computing tracking with PoseFlow')

        img_dir = dirOut2 + '/vis'
        alphapose_json = alphajson[0]

        cmd = [
            'python3', 'PoseFlow/tracker-general.py',
            '--imgdir', img_dir,
            '--in_json', alphapose_json,
            '--out_json', out_json,
            # '--visdir', out_dir,  # Uncomment this to visualize PoseFlow tracks.
        ]
        print('Running: {}'.format(' '.join(cmd)))
        curr_dir = os.getcwd()

        ret = subprocess.call(cmd)
        if ret != 0:
            print('Issue running PoseFlow. Please make sure you can run the above '
                  'command from the commandline.')
            exit(ret)

        print('PoseFlow successfully ran!')
        print('----------')
        newKps = splitFileAlpha(out_json, nFrame, track_id)
    return newKps

if __name__ == '__main__':

    #main_dir = '/home/nayari/projects/SoccerKicks/'
    main_dir = '/home/nayari/projects/youtube_data/output_video/AlphaPose/'

    videos_dir = main_dir + 'Original_videos/'

    dir = main_dir + '/Rendered/'

    action = 'Freekick/' #Penalty

    name_file = 'Y1'

    track_id = 2

    ouput_dir = main_dir + name_file
    alphapose_json = [ouput_dir + '/alphapose-results.json']

    im_paths_alpha = len(sorted(glob(osp.join(ouput_dir + '/vis', "*.jpg"))))
    print ('Number of 2D AlphaPose output frames', im_paths_alpha, '\n')

    nFrame = im_paths_alpha

    keyAlpha(ouput_dir, alphapose_json, nFrame, track_id)
