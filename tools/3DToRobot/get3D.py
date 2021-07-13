"""#######################
    In this code the 3D joint location SMPL is computed to:

    *Absolute joint location (Global)

    Given:
    'poses'  Nx24x3x3 and is a rotation matrices corresponding to the SMPL joint
     N = number of frames
"""
import os
import json
import ipdb
import pickle as pkl
import numpy as np

from tools import *
from drawSkeleton import *

SMPL_MODEL_PATH = ('model/neutral_smpl_with_cocoplustoesankles_reg.pkl')

class SMPL(object):
    "This code was build on human_dynamics/src/tf_smpl/batch_smpl.py"

    def __init__(self, pkl_path, joint_type='cocoplus'):

        # -- Load SMPL params --
        smpl_data = pkl.load(open(pkl_path, 'rb'), encoding = 'latin1')

        self.smpl_data = smpl_data

        self.shapedirs = np.array(smpl_data['shapedirs'], dtype="float32")

        self.posedirs = np.array(smpl_data['posedirs'], dtype="float32")

        self.v_template = np.array(smpl_data['v_template'])
        self.v_template = np.expand_dims(self.v_template, axis = 0)

        self.J_regressor = np.array(smpl_data['J_regressor'].toarray(), dtype="float32")

        self.weights = np.array(smpl_data['weights'], dtype="float32")

        self.faces =np.array(smpl_data['f'].astype(np.int32))

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']

        self.parents = list(self.kintree_table[0].tolist())
        #parents [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        self.num_joints = len(self.parents)  # 24
        print('SMPL model has %s joints' % self.num_joints)

    def __call__(self, gl_pose, beta):

        #Given the rotation matrix - gl_pose (Nx24x3x3)
        batch_size = beta.shape[0]
        print('There are %s Frames' % batch_size)

        # Take out the first rotmat (global rotation) - (Nx3x3)
        #root_rotation = np.resize(gl_pose[:, :9], (batch_size, 3, 3))
        root_rotation = gl_pose[:, 0, :, :]

        # Take out the remaining rotmats (23 joints) (Nx23x3x3)
        roots = gl_pose[:, 1 :, :, :]

        # Subtract the identity pose matrix from the current pose matrix | np.eye --> Construct an identity matrix
        posemap = np.reshape(roots - np.eye(3, dtype=roots.dtype), [-1, 207]) #shape --> Nx207

        # v_template shape = (1x6890x3); shapedirs shapes (6890x3x10); beta shape = (Nx10)
        # Result in vertposes shape = (Nx6890x3)
        vertshape = self.v_template + np.transpose(np.matmul(self.shapedirs, np.transpose(beta, (1, 0))), (2, 0, 1))

        # self.posedirs shape (6890x3x207) | Result in vertposes shape = (Nx6890x3)
        vertposes = vertshape + np.transpose(np.matmul(self.posedirs, np.transpose(posemap,(1, 0))), (2, 0, 1))

        #Infer shape-dependent joint locations. (Nx24x3)
        J = np.matmul(self.J_regressor, vertshape)
        """
        Computes absolute joint locations (global joint location) given pose.
        """
        #global coordinates NX24X3
        Jgl, A = batch_global_rigid_transformation(
            gl_pose, J, self.parents)

        root_joint = Jgl[:, 0]

        """
        Global joint location - 4d homogenous coordiantes
        """
        #Converts an array in 3d coordinates to 4d homogenous coordiantes (N x 24 x 4)
        global_joints = with_ones(Jgl)


        return Jgl, root_rotation, root_joint

class main(SMPL):

    def __init__(self, preds_path, save_dir):

        with open(preds_path, 'rb') as f:
            preds = pkl.load(f)
        f.close()

        all_kps = preds['kps']
        joints = preds ['joints'] #image coordinates
        poses= preds['poses']
        cams= preds['cams']
        verts = preds['verts']
        shapes = preds['shapes']
        omegas = preds['omegas']

        smpl = SMPL(SMPL_MODEL_PATH)
        Jgl, root_rotation, root_joint = smpl(poses, shapes)

        """
        Draw Skeleton
        """
        #frame = 98
        #make_image(jgl, save_dir, frame)





if __name__ == '__main__':

    main_dir = '/home/nayari/projects/SoccerKicks'

    dir = main_dir + '/Rendered/'

    action = 'Freekick/' #Penalty

    name_file = '22_freekick' #

    ouput_dir = dir + action + name_file + '/annotations/'

    preds_file = 'hmmr_output.pkl'

    input_alpha_hmmr =  dir + action + name_file + '/hmmr_output/'+ preds_file

    input_open_hmmr = dir + action + name_file + '/hmmr_output_openpose/'+ preds_file

    save_dir = dir + action + name_file + '/frame.jpg'

    main(input_alpha_hmmr, save_dir)
