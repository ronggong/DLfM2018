import numpy as np
from sklearn.metrics import average_precision_score


def ground_truth_matrix(y_test):
    """
    ground truth mat
    :param y_test:
    :return:
    """
    sample_num = len(y_test)

    gt_matrix = np.zeros((sample_num, sample_num))

    for ii in range(sample_num-1):
        for jj in range(ii+1, sample_num):
            if y_test[ii] == y_test[jj]:
                gt_matrix[ii, jj] = 1.0
            else:
                gt_matrix[ii, jj] = 0.0
    return gt_matrix


def eval_embeddings(dist_mat, gt_mat):
    """
    average precision score
    :param dist_mat:
    :param gt_mat:
    :return:
    """
    assert dist_mat.shape == gt_mat.shape
    sample_num = dist_mat.shape[0]
    iu1 = np.triu_indices(sample_num, 1) # trim the upper mat

    print(len(gt_mat[iu1][gt_mat[iu1]==0]))
    print(len(gt_mat[iu1]))
    print(dist_mat[iu1])
    ap = average_precision_score(y_true=np.abs(gt_mat[iu1]), y_score=np.abs(dist_mat[iu1]), average='weighted')
    return ap


def eval_embeddings_no_trim(dist_mat, gt_mat):
    """
    average precision score
    :param dist_mat:
    :param gt_mat:
    :return:
    """
    assert dist_mat.shape == gt_mat.shape
    ap = average_precision_score(y_true=np.squeeze(np.abs(gt_mat)),
                                 y_score=np.squeeze(np.abs(dist_mat)),
                                 average='weighted')
    return ap
