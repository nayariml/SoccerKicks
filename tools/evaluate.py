###
### Adaptaded from https://github.com/aymenmir1/3dpw-eval/
###
import os
import numpy as np
import pickle as pkl
import glob
import cv2
import sys

PCK_THRESH = 50.0
AUC_MIN = 0.0
AUC_MAX = 200.0

NR_JOINTS = 25

IMPORTANT_JOINTS = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,23,24])

def compute_auc(xpts, ypts):
    """
    Calculates the AUC.
    :param xpts: Points on the X axis - the threshold values
    :param ypts: Points on the Y axis - the pck value for that threshold
    :return: The AUC value computed by integrating over pck values for all thresholds
    """
    a = np.min(xpts)
    b = np.max(xpts)
    from scipy import integrate
    myfun = lambda x: np.interp(x, xpts, ypts)
    auc = integrate.quad(myfun, a, b)[0]
    return auc


def compute_pck(errors, THRESHOLD):
    """
    Computes Percentage-Correct Keypoints
    :param errors: N x 12 x 1
    :param THRESHOLD: Threshold value used for PCK
    :return: the final PCK value
    """
    errors_pck = errors <= THRESHOLD
    errors_pck = np.mean(errors_pck, axis=1)
    return np.mean(errors_pck)


def compute_errors(preds3d, gt3ds):

    errors, errors_pa, errors_pck = [], [], []

    proc_rot = []

    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds3d)):

        # Compute erros
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        # Joint errors for PCK Calculation
        joint_error_maj = joint_error[IMPORTANT_JOINTS]
        errors_pck.append(joint_error_maj)

    return np.stack(errors_pck, 0)


def compute_errors_function(jp_pred, jp_gt):

    if len(jp_gt) > len(jp_pred):
        jp_gt = jp_gt[:(len(jp_pred))]
        print('The evaluation will be done by the end of the openpose size.', len(jp_gt))
    else:
        jp_pred = jp_pred[:(len(jp_gt))]
        print('The evaluation will be done by the end of the alphapose size.', len(jp_gt))


    # Check if the predicted and GT joints have the same number
    assert jp_pred.shape == jp_gt.shape

    # If there are submitted joint predictions
    if not jp_pred.shape == (0,):

        # Joint errors and procrustes matrices
        errors_pck = compute_errors(jp_pred * 1000., jp_gt * 1000.)

        # PCK value
        pck_final = compute_pck(errors_pck, PCK_THRESH) * 100.

        # AUC value
        auc_range = np.arange(AUC_MIN, AUC_MAX)
        pck_aucs = []
        for pck_thresh_ in auc_range:
            err_pck_tem = compute_pck(errors_pck, pck_thresh_)
            pck_aucs.append(err_pck_tem)

        auc_final = compute_auc(auc_range / auc_range.max(), pck_aucs)

    errs = {
        'PCK': pck_final,
    }

    str = ''
    for err in errs.keys():
        if not errs[err] == np.inf:
            str = str + err + ': {}\n'.format(errs[err])

    return str, errs
