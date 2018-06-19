import numpy as np
import h5py
import itertools

from keras.utils.np_utils import to_categorical


def shuffleFilenamesLabelsInUnison(filenames, labels):
    assert len(filenames) == len(labels)
    p=np.random.permutation(len(filenames))
    return filenames[p], labels[p]


def generator(path_feature_data,
              indices,
              number_of_batches,
              file_size,
              input_shape,
              labels=None,
              shuffle=True,
              multi_inputs=False,
              channel=1):

    # print(len(filenames))
    # print(path_feature_data)
    f = h5py.File(path_feature_data, 'r')
    indices_copy = np.array(indices[:], np.int64)

    if labels is not None:
        labels_copy = np.copy(labels)
        labels_copy = to_categorical(labels_copy)
    else:
        labels_copy = np.zeros((len(indices_copy), ))

    counter = 0

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        batch_indices = indices_copy[idx_start:idx_end]
        index_sort = np.argsort(batch_indices)

        y_batch_tensor = labels_copy[idx_start:idx_end][index_sort]

        if channel == 1:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort],:,:]
        else:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort], :, :, :]
        if channel == 1:
            X_batch_tensor = np.expand_dims(X_batch_tensor, axis=1)

        counter += 1

        if multi_inputs:
            yield [X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor], y_batch_tensor
        else:
            yield X_batch_tensor, y_batch_tensor

        if counter >= number_of_batches:
            counter = 0
            if shuffle:
                indices_copy, labels_copy = shuffleFilenamesLabelsInUnison(indices_copy, labels_copy)


def sort_feature_by_seq_length(list_feature, labels, batch_size):
    # sort the list_feature and the labels by the length
    list_len = [len(l) for l in list_feature]
    idx_sort = np.argsort(list_len)
    list_feature_sorted = [list_feature[ii] for ii in idx_sort]
    if isinstance(labels, list):
        labels_sorted = [labels[ii] for ii in idx_sort]
    else:
        labels_sorted = labels[idx_sort]

    iter_times = int(np.ceil(len(list_feature_sorted) / float(batch_size)))

    return list_feature_sorted, labels_sorted, iter_times


def build_y_batch(batch_size, labels_batch):
    """reorganize y batch to a 2 dim list"""
    y_phn, y_pro = np.zeros((batch_size, len(labels_batch[0][0]))), np.zeros((batch_size, len(labels_batch[0][1])))
    for ii, (phn, pro) in enumerate(labels_batch):
        y_phn[ii, :] = phn
        y_pro[ii, :] = pro
    y_batch = [y_phn, y_pro]
    return y_batch


def batch_grouping(list_feature_sorted, labels_sorted, iter_times, batch_size):
    """
    group the features and labels into batches,
    each feature batch has more or less the sequence with the similar length
    :param list_feature:
    :param labels:
    :param batch_size:
    :return:
    """

    # aggregate the iter_times-1 batch
    list_X_batch = []
    list_y_batch = []
    for ii in range(iter_times-1):
        max_len_in_batch = list_feature_sorted[(ii+1)*batch_size-1].shape[0]
        feature_dim_in_batch = list_feature_sorted[(ii+1)*batch_size-1].shape[1]

        X_batch = np.zeros((batch_size, max_len_in_batch, feature_dim_in_batch), dtype='float32')
        if isinstance(labels_sorted, list):
            y_batch = build_y_batch(batch_size, labels_sorted[ii * batch_size: (ii + 1) * batch_size])
        else:
            y_batch = labels_sorted[ii * batch_size: (ii + 1) * batch_size, :]

        # print(ii*batch_size, (ii+1)*batch_size)
        for jj in range(ii*batch_size, (ii+1)*batch_size):
            X_batch[jj-ii*batch_size, :len(list_feature_sorted[jj]), :] = list_feature_sorted[jj]

        list_X_batch.append(X_batch)
        list_y_batch.append(y_batch)

    # aggregate the last batch
    max_len_in_batch = list_feature_sorted[-1].shape[0]
    feature_dim_in_batch = list_feature_sorted[-1].shape[1]

    X_batch = np.zeros((batch_size, max_len_in_batch, feature_dim_in_batch), dtype='float32')
    if isinstance(labels_sorted, list):
        y_batch = build_y_batch(batch_size, labels_sorted[-batch_size:])

    else:
        y_batch = labels_sorted[-batch_size:, :]

    for jj in range(len(list_feature_sorted)-batch_size, len(list_feature_sorted)):
        X_batch[jj-(len(list_feature_sorted)-batch_size), :len(list_feature_sorted[jj]), :] = list_feature_sorted[jj]

    list_X_batch.append(X_batch)
    list_y_batch.append(y_batch)

    return list_X_batch, list_y_batch


def shuffleListBatch(list_X_batch, list_y_batch):
    assert len(list_X_batch) == len(list_y_batch)
    p = np.random.permutation(len(list_X_batch))
    list_X_batch = [list_X_batch[ii] for ii in p]
    list_y_batch = [list_y_batch[ii] for ii in p]

    for ii_batch in range(len(list_X_batch)):
        p = np.random.permutation(list_X_batch[ii_batch].shape[0])
        list_X_batch[ii_batch] = list_X_batch[ii_batch][p, :, :]
        if len(list_y_batch[ii_batch]) == 2:
            list_y_batch[ii_batch] = [list_y_batch[ii_batch][0][p, :], list_y_batch[ii_batch][1][p, :]]
        else:
            list_y_batch[ii_batch] = list_y_batch[ii_batch][p, :]

    return list_X_batch, list_y_batch


def generator_batch_group(list_X_batch,
                          list_y_batch,
                          iter_times,
                          shuffle=True):

    ii = 0
    while True:
        yield list_X_batch[ii], list_y_batch[ii]

        ii += 1
        if ii >= len(list_X_batch):
            ii = 0
            if shuffle:
                list_X_batch, list_y_batch = shuffleListBatch(list_X_batch=list_X_batch,
                                                              list_y_batch=list_y_batch)
