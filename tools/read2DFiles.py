#Implemented by @nayariml 
import os
import csv
import math
import json
import numpy as np

from plot import TwoDplot
from split_kps import keyAlpha
from matplotlib import pyplot as plt
from glob import glob
#import os.path as osp
from os.path import join, exists, basename
from os import makedirs, system


im_joints = {} # 21 joints : Same struct of OpenPose exceeded with head joint of AlphaPose
im_joints = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10:"RKnee",
    11:"RAnkle",
    12:"LHip",
    13:"LKnee",
    14:"LAnkle",
    15:"REye",
    16:"LEye",
    17:"REar",
    18: "LEar",
    19:"LBigToe",
    20:"LSmallToe",
    21:"LHeel",
    22:"RBigToe",
    23:"RSmallToe",
    24:"RHeel",
    25: "Head"}

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory " , dir_path ,  " Created ")
    else:
        print("Directory " , dir_path ,  " already exists")

def check_points(all_kps):

    error2d = {}; count_kps = 0; kps = []; kp_name = []

    for i in range(25):
        if np.any(all_kps[i] == 0):
            count_kps += 1
            kps = [im_joints[i], all_kps[i]]
            kp_name.append(kps)
            kps = []

    error2d = {'count_error': count_kps, 'error2d': kp_name}

    return error2d

def split_joints(all_kps, nFrame, saveDir, alpha):

    new_kps =[]; kpa = []; save_file = []

    for i, kp in enumerate(all_kps):
        new_kps =[]
        #if np.all(kp == None):
        #    kp = np.ones(78).reshape(26,3)
        new_kps.append(kp[:,:2])

        if alpha:
            kps = ordering_alpha(new_kps[0])
            kpa.append(kps)
            error2d = check_points(kpa[i])
            save_file.append([i,error2d['count_error'], error2d['error2d'], kpa[i][0], kpa[i][1], kpa[i][2],
                    kpa[i][3], kpa[i][4], kpa[i][5], kpa[i][6],
                    kpa[i][7], kpa[i][8],kpa[i][9], kpa[i][10],
                    kpa[i][11], kpa[i][12], kpa[i][13], kpa[i][14],
                    kpa[i][19], kpa[i][20], kpa[i][21], kpa[i][22], kpa[i][23],
                    kpa[i][24], kpa[i][17]])

        if not alpha:
            kps0 = 0
            error2d = check_points(new_kps[0])
            kpa = new_kps[0]
            save_file.append([i, error2d['count_error'], error2d['error2d'], kpa[0], kpa[1], kpa[2],
                    kpa[3], kpa[4], kpa[5], kpa[6],
                    kpa[7], kpa[8],kpa[9], kpa[10],
                    kpa[11], kpa[12], kpa[13], kpa[14],
                    kpa[19], kpa[20], kpa[21], kpa[22], kpa[23],
                    kpa[24], kps0])
    print('Saving 2D keypoints in:', saveDir)
    print('\n')
    save_2d_kps(saveDir,save_file)
    return

def ordering_alpha(kps):
    #print(kps)
    kp = []
    kp = [kps[0],kps[18],kps[6],kps[8],kps[10],kps[5],kps[7],kps[9],kps[19],kps[12],kps[14],kps[16], kps[11], kps[13],
    kps[15],kps[2],kps[1],kps[4],kps[3],kps[20],kps[22],kps[24],kps[21],kps[23],kps[25], kps[17]]
    return kp

def save_2d_kps(saveDir, files):

    with open(saveDir, "a") as csvfile:
        file_is_empty = os.stat(saveDir).st_size == 0
        headers = ['Frame nº', 'Count_error', 'error', im_joints[0], im_joints[1], im_joints[2],
                im_joints[3], im_joints[4], im_joints[5], im_joints[6],
                im_joints[7], im_joints[8],im_joints[9], im_joints[10],
                im_joints[11], im_joints[12], im_joints[13], im_joints[14],
                im_joints[19], im_joints[20], im_joints[21], im_joints[22], im_joints[23],
                im_joints[24], im_joints[25]]
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(files)

    return

def keyOpen(kps):
    min_kp_count = 15
    pVis = []; counts_list = []

    for kp in kps:
        vis = kp[:, 2] > 0 #For all the positions (x, y) > 0 = True
        if np.sum(vis) >= min_kp_count: #Count if the sum of points are more than min acceptable
            pVis.append(kp)
            counts_list.append(len(pVis))

    sort_idx = np.argsort(counts_list)[::-1]
    kps_list_sorted = []
    for sort_id in sort_idx:
        kps_list_sorted.append(pVis[sort_id])

    return kps_list_sorted

def read_json(json_path): #Openpose
    with open(json_path) as f1:
        data = json.load(f1)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps

class digest_sys():

    def __init__(self,track_id, input_dir, input_alpha, ouput_dir, main_dir, name_file):

        self.track_id = track_id

        self.output_dir = main_dir
        self.name_file = name_file

        self.frames_dir = input_dir + '/video_frames'

        self.dir_openpose = input_dir + '/OpenPose_output'
        self.dir_alphapose = input_dir + '/AlphaPose_output'

        self.dir_alphapose_hmmr = input_alpha + '/video_out'
        self.dir_alphapose_vis = self.dir_alphapose + '/vis'

        self.saveDir = ouput_dir +'2D_pose_keypoints' # Inside Annotation
        mkdir(self.saveDir)

        self.saveDir_open = self.saveDir + '/OpenPose_2D_kps.csv'
        self.saveDir_alpha = self.saveDir + '/AlphaPose_2D_kps.csv'

        if not os.path.exists(self.dir_alphapose_hmmr):

            # Get all the images of AlphaPose
            im_paths_alpha = len(sorted(glob(join(self.dir_alphapose_vis, "*.jpg"))))
            print ('Number of 2D rendered frames AlphaPose', im_paths_alpha, '\n')
            self.nFrame_alpha = im_paths_alpha
        else:
            # Get all the images of AlphaPose
            im_paths_alpha = len(sorted(glob(join(self.dir_alphapose_hmmr, "*.png"))))
            print ('Number of 2D hmmr- AlphaPose output frames', im_paths_alpha, '\n')
            self.nFrame_alpha = im_paths_alpha

        if os.path.exists(self.dir_openpose):
            print('Reading %s' % self.dir_openpose)
            json_paths = sorted(glob(join(self.dir_openpose, "*.json")))#Openpose
            im_paths_open = len(sorted(glob(join(self.dir_openpose, "*.jpg"))))# Get all the images of openpose
            print ('Number of 2D rendered frame - OpenPose', im_paths_open, '\n')
            self.nFrame_open = im_paths_open

            all_kps_open = []
            for i, json_path in enumerate(json_paths): #Reading Openpose json files
                kps = read_json(json_path)
                all_kps_open.append(kps)
            print('OpenPose json files: ok')

            nbox = []; track_id_ = 0; count = 0
            for fls in all_kps_open:
                box = keyOpen(fls)
                count += 1
                if len(box) < 1:
                    print('OpenPose - Person does not have the minimum points on the frame nº - ', count - 1)
                    break #Break in the first occurrence
                track_id_ = min(track_id_, len(box) - 1)
                kps = box[track_id_]
                nbox.append(kps)
            all_kps_open = nbox
            print('Len OpenPose all_kps', len(all_kps_open), '\n')
            split_joints(all_kps_open, self.nFrame_open, self.saveDir_open, alpha=False)

            if not os.path.exists(self.dir_alphapose):
                self.get_kps_open(all_kps_open)
                print('Done!')

        if os.path.exists(self.dir_alphapose):
            print('Reading %s' % self.dir_alphapose , '\n')
            json_paths2 = sorted(glob(join(self.dir_alphapose, 'alphapose-results.json')))#AlphaPose
            if os.path.exists(self.dir_openpose):
                all_kps_alpha = keyAlpha(self.dir_alphapose, json_paths2, self.nFrame_alpha, self.track_id )
                print('Len AlphaPose all_kps', len(all_kps_alpha), '\n')

            else:
                all_kps_alpha = keyAlpha(self.dir_alphapose, json_paths2, self.nFrame_alpha, self.track_id )
                print('Len AlphaPose all_kps', len(all_kps_alpha), '\n')
                split_joints(all_kps_alpha, self.nFrame_alpha, self.saveDir_alpha, alpha=True)
                self.get_kps_alpha(all_kps_alpha)
                print('Done!')


        if os.path.exists(self.dir_alphapose) and os.path.exists(self.dir_openpose):

            if self.nFrame_open == len(all_kps_open) and self.nFrame_alpha ==len(all_kps_alpha):
                print("OpenPose and AlphaPose have only one person detected.", '\n')
                print("Or have the same number of Frames rendered.", '\n')
                split_joints(all_kps_alpha, self.nFrame_open, self.saveDir_alpha, alpha=True)
                self.clean_detections(all_kps_open, all_kps_alpha)

            elif self.nFrame_alpha < self.nFrame_open:
                new_open_kps = all_kps_open[:self.nFrame_alpha]
                if self.nFrame_alpha == len(new_open_kps) and self.nFrame_alpha == len(all_kps_alpha) :
                    split_joints(all_kps_alpha[:self.nFrame_alpha], self.nFrame_alpha, self.saveDir_alpha, alpha=True)
                    print('The metches will be done by the end of the AlphaPose size frames.', len(new_open_kps))
                    self.clean_detections(new_open_kps, all_kps_alpha)
                else:
                    print('Could not possible do the matching')
                    Exception

            elif self.nFrame_open < len(all_kps_alpha):
                new_alpha_kps = all_kps_alpha[:self.nFrame_open]
                print('The metches will be done by the end of the OpenPose size frames.', len(new_alpha_kps))
                if self.nFrame_open == len(new_alpha_kps):
                    split_joints(new_alpha_kps, self.nFrame_open, self.saveDir_alpha, alpha=True)
                    self.clean_detections(all_kps_open, new_alpha_kps)
                else:
                    print('Could not find the player in all frames in AlphaPose')
                    Exception


    def clean_detections(self, all_kps, all_kps2):

        X_box1 = []; X_box2 = []; bbox_x= []; calcF = []; time = []; SAVE = []; SAVEP = []; joins = []
        Y_box1 = []; Y_box2 = []; bbox_y = [];calcF2 = []; t = 0; porc = []; porc2 = []; head = []

        #if len(all_kps) == len(all_kps2):
        print('Starting the comparison:', len(all_kps))

        for kp1 in all_kps:
            viss, p = getPercent(kp1)
            vis = np.squeeze(viss)
            porc.append([vis, p])
            x, y = get_bbox(kp1)
            X_box1.append(x)#25
            Y_box1.append(y)
            time.append(t)
            t += 1

        for kp2 in all_kps2:
            x2, y2 = get_bbox(kp2)
            x2new, y2new = getAlphavector(x2,y2)
            bbox_x.append(x2new)
            bbox_y.append(y2new)

        xx = np.array(bbox_x).reshape(-1,1)
        yy = np.array(bbox_y).reshape(-1,1)
        joint = np.concatenate([xx, yy], axis = 1)
        viss, p = getPercent2(joint)
        porc2.append([viss, p])
        pporc = np.squeeze(porc2)

        print('LEN_BOX: OpenPosen - AlphaPose (x, x, y, y)', len(X_box1), len(bbox_x), len(Y_box1), len(bbox_y))
        #print('box', bbox_x)
        euclid_mean = []; porc_mean_op = []; porc_mean_al = []; al_kps_mean = []; op_kps_mean = []; SAVE_KPS = []
        for f in range (self.nFrame_open):
            calc1 = eucDist(X_box1[f], bbox_x[f],Y_box1[f],bbox_y[f])
            calcF.append(calc1)

            euclid_mean.append(np.mean(calc1))

            op_kps_mean.append(int((np.mean(porc[f][0]))))
            porc_mean_op.append(int(np.mean(porc[f][1])))

            al_kps_mean.append(int(np.mean(pporc[0][f])))
            porc_mean_al.append(int(np.mean(pporc[1][f])))



            SAVE.append([f, calcF[f][0], calcF[f][1], calcF[f][2],
                    calcF[f][3], calcF[f][4], calcF[f][5], calcF[f][6],
                    calcF[f][7], calcF[f][8],calcF[f][9], calcF[f][10],
                    calcF[f][11], calcF[f][12], calcF[f][13], calcF[f][14],
                    calcF[f][19], calcF[f][20], calcF[f][21], calcF[f][22], calcF[f][23],
                    calcF[f][24]])


            SAVEP.append([f, int(porc[f][0]), int(porc[f][1]), int(pporc[0][f]), int(pporc[1][f])])

        save(SAVE,  self.saveDir)
        saveP(SAVEP,  self.saveDir)
        #TwoDplot(calcF, self.nFrame_open, time, X_box1, bbox_x, Y_box1, bbox_y, self.saveDir)
        SAVE_KPS.append([self.name_file, self.nFrame_open, np.mean(op_kps_mean), np.mean(porc_mean_op), np.mean(al_kps_mean), np.mean(porc_mean_al), np.mean(euclid_mean), None,
        np.mean(SAVE[0][1]), np.mean(SAVE[0][2]), np.mean(SAVE[0][3]), np.mean(SAVE[0][4]), np.mean(SAVE[0][5]), np.mean(SAVE[0][6]), np.mean(SAVE[0][7]), np.mean(SAVE[0][8]),
        np.mean(SAVE[0][9]), np.mean(SAVE[0][10]), np.mean(SAVE[0][11]), np.mean(SAVE[0][12]), np.mean(SAVE[0][13]), np.mean(SAVE[0][14]), np.mean(SAVE[0][15]), np.mean(SAVE[0][16]),
        np.mean(SAVE[0][17]), np.mean(SAVE[0][18]), np.mean(SAVE[0][19]), np.mean(SAVE[0][20]), np.mean(SAVE[0][21])])

        save_mean(SAVE_KPS, self.output_dir)

    def get_kps_open(self, all_kps):

        porc = []; X_box1 = []; Y_box1 = []; op_kps_mean = []; porc_mean_op = []; SAVE_KPS = []
        for f, kp1 in enumerate(all_kps):
            viss, p = getPercent(kp1)
            vis = np.squeeze(viss)
            porc.append([vis, p])
            x, y = get_bbox(kp1)
            X_box1.append(x)#25
            Y_box1.append(y)

            op_kps_mean.append(int((np.mean(porc[f][0]))))
            porc_mean_op.append(int(np.mean(porc[f][1])))

        SAVE_KPS.append([self.name_file, self.nFrame_open, np.mean(op_kps_mean), np.mean(porc_mean_op), None, None, None])
        save_mean(SAVE_KPS, self.output_dir)
        return

    def get_kps_alpha(self, all_kps2):

        bbox_x = []; bbox_y = []; porc2 = []; al_kps_mean = []; porc_mean_al = []; SAVE_KPS = []
        for f, kp2 in enumerate(all_kps2):
            x2, y2 = get_bbox(kp2)
            x2new, y2new = getAlphavector(x2,y2)
            bbox_x.append(x2new)
            bbox_y.append(y2new)

        xx = np.array(bbox_x).reshape(-1,1)
        yy = np.array(bbox_y).reshape(-1,1)
        joint = np.concatenate([xx, yy], axis = 1)
        viss, p = getPercent2(joint)
        porc2.append([viss, p])
        pporc = np.squeeze(porc2)

        al_kps_mean.append(int(np.mean(pporc[0][f])))
        porc_mean_al.append(int(np.mean(pporc[1][f])))
        SAVE_KPS.append([self.name_file, self.nFrame_alpha, None, None, np.mean(al_kps_mean), np.mean(porc_mean_al), None])
        save_mean(SAVE_KPS, self.output_dir)
        return

def save(files, saveDir):
    mkdir(saveDir)
    with open(saveDir + '/Euclidean_distance.csv', "a") as csvfile:
        file_is_empty = os.stat(saveDir +'/Euclidean_distance.csv').st_size == 0
        headers = ['Frames_N', im_joints[0], im_joints[1], im_joints[2],
                im_joints[3], im_joints[4], im_joints[5], im_joints[6],
                im_joints[7], im_joints[8],im_joints[9], im_joints[10],
                im_joints[11], im_joints[12], im_joints[13], im_joints[14],
                im_joints[19], im_joints[20], im_joints[21], im_joints[22], im_joints[23],
                im_joints[24]]
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(files)
    return

def saveP(files,  saveDir):
    with open(saveDir + '/percentage_kps.csv', "a") as csvfile:
        file_is_empty = os.stat(saveDir +'/percentage_kps.csv').st_size == 0
        headers = ['Frames_N', 'Openpose_Num', 'Openpose_Porc_%','AlphaPose_Num', 'AlphaPose_Porc_%']
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(files)
    return

def save_mean(files, output_dir):
    with open(output_dir + '/2D_kps_info.csv', "a") as csvfile:
        file_is_empty = os.stat(output_dir +'/2D_kps_info.csv').st_size == 0
        headers = ['name_file','Frames_N', 'Average_OpenPose_kps', 'Average_Openpose_Porc_%','Average_AlphaPose_kps', 'Average_AlphaPose_Porc_%', 'Average_euclidean_distance_all points', 'Average_euclidean_distance_each_point:',
                im_joints[0], im_joints[1], im_joints[2],
                im_joints[3], im_joints[4], im_joints[5], im_joints[6],
                im_joints[7], im_joints[8],im_joints[9], im_joints[10],
                im_joints[11], im_joints[12], im_joints[13], im_joints[14],
                im_joints[19], im_joints[20], im_joints[21], im_joints[22], im_joints[23],
                im_joints[24]]
        writer = csv.writer(csvfile)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerows(files)
    return
def get_bbox(kp):
    #if np.all(kp == None):
    #    kp = np.ones(78).reshape(26,3)
    kp_len = len(kp)
    #kp_len = 25
    x = []
    y = []
    for i in range(kp_len):

        x.append(kp[i][0])
        y.append(kp[i][1])
    return x, y

def getAlphavector(x,y): #Ordering AlphaVector to OpenPose

    xnew = [x[0],x[18],x[6],x[8],x[10],x[5],x[7],x[9],x[19],x[12],x[14],x[16], x[11], x[13],
    x[15],x[2],x[1],x[4],x[3],x[20],x[22],x[24],x[21],x[23],x[25]]

    ynew = [y[0],y[18],y[6],y[8],y[10],y[5],y[7],y[9],y[19],y[12],y[14],y[16], y[11], y[13],
    y[15],y[2],y[1],y[4],y[3],y[20],y[22],y[24],y[21],y[23],y[25]]

    #head = [x[17],y[17]]

    if len(xnew) == len(ynew):
        return xnew, ynew
    else:
        print('Something wrong!')
        Exception

def getPercent(kps):

    viss = []; porc = []
    vis = kps[:, 2] > 0 #For all the positions (x, y) > 0 = True
    sumP = np.sum(vis)
    p = (sumP / 25)*100
    viss.append(sumP)
    porc.append(p)
    #print("Porcentage for each frame", p)
    return viss, p

def getPercent2(kps):
    viss = []; porc = []; sumP =[]
    ii = 0
    for kp in kps:
        vis = kp > 0 #For all the positions (x, y) > 0 = True
        sumP.append(np.sum(vis))
        ii += 1
        if ii == 25:
            ii = 0
            viss.append(np.sum(sumP)/2)
            summ = np.sum(sumP)/2
            p = (summ / 25)*100
            porc.append(p)
            #print("Porcentage for each frame", p)
            sumP = []
    return viss, porc

def eucDist(x1, x2, y1, y2):
    calcd = []

    def cald(x1, x2, y1, y2):
        #print(x1); print(x2); print(y1); print(y2)
        distance = 0; distance2 = 0
        distance += pow((x1 - x2), 2)
        distance2 += pow((y1 - y2), 2)
        dist1 = math.sqrt(distance + distance2)
        return dist1
    for ii in range(25):#EACH KEYPOINTS
        dist1 = cald(x1[ii], x2[ii], y1[ii], y2[ii])
        calcd.append(dist1)
    return calcd


if __name__ == '__main__':

    main_dir = '/home/nayari/projects/SoccerKicks/'

    videos_dir = main_dir + 'Original_videos/'

    dir = main_dir + '/Rendered/'

    action = 'Penalty/' #Penalty

    name_file = '17_penalty'

    track_id = 2

    input_dir = dir + action + name_file

    ouput_dir = dir + action + name_file + '/annotations/'

    digest_sys(track_id, input_dir, ouput_dir)
