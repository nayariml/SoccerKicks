import numpy as np


def with_ones(data):
    """
    Converts an array in 3d coordinates to 4d homogenous coordiantes

    data: array of shape N x J x 3
    return: array of shape N x J x 4 where the extra dimension is filled with ones
    """
    extrad = np.ones((data.shape[0], data.shape[1], 1))
    new_data = np.concatenate((data, extrad), axis=2)
    return new_data

def batch_global_rigid_transformation(Rs, Js, parent):
    "Adapted from: human_dynamics/src/tf_smpl/batch_smpl.py"
    """
    Computes absolute joint locations given pose.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index
    """
    N = Rs.shape[0]

    root_rotation = Rs[:, 0, :, :] #(NX3X3)

    # Js -> N x 24 x 3 x 1
    Js = np.expand_dims(Js, -1)

    def make_A(R, t, name=None):
        # Rs is N x 3 x 3, ts is N x 3 x 1

        R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]])
        t_homo = np.concatenate([t, np.ones([N, 1, 1])], 1)
        return np.concatenate([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, len(parent)):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = np.matmul(
            results[parent[i]], A_here)
        results.append(res_here)

    results = np.stack(results, axis=1)

    new_J = results[:, :, :3, 3]

    Js_w0 = np.concatenate([Js, np.zeros([N, 24, 1, 1])], 2)
    init_bone = np.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = np.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
    A = results - init_bone

    return new_J, A
