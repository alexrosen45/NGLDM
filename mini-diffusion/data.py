"""
This data loader script was taking from assignment 3 of CSC311 Fall 2023 at the University of Toronto
"""
import zipfile
import numpy as np
import os


np.random.seed(1746)
PREFIX = "digit_"
TEST_STEM = "test_"
TRAIN_STEM = "train_"


def check_and_extract_zipfile(filename, data_dir):
    if os.path.isdir(data_dir) and not os.listdir(data_dir):
        pass
    else:
        zip_f = zipfile.ZipFile(filename, 'r')
        zip_f.extractall(data_dir)
        zip_f.close()


def load_data(data_dir, stem):
    data = []
    labels = []
    for i in range(0, 10):
        path = os.path.join(data_dir, PREFIX + stem + str(i) + ".txt")
        digits = np.loadtxt(path, delimiter=',')
        digit_count = digits.shape[0]
        data.append(digits)
        labels.append(np.ones(digit_count) * i)
    data, labels = np.array(data), np.array(labels)
    data = np.reshape(data, (-1, 64))
    labels = np.reshape(labels, (-1))
    return data, labels


def load_all_data(data_dir, shuffle=True):
    if not os.path.isdir(data_dir):
        raise OSError('Data directory {} does not exist. Try "load_all_data_from_zip" function first.'.format(data_dir))

    train_data, train_labels = load_data(data_dir, TRAIN_STEM)
    test_data, test_labels = load_data(data_dir, TEST_STEM)

    if shuffle:
        train_indices = np.random.permutation(train_data.shape[0])
        test_indices = np.random.permutation(test_data.shape[0])
        train_data, train_labels = train_data[train_indices], train_labels[train_indices]
        test_data, test_labels = test_data[test_indices], test_labels[test_indices]

    return train_data, train_labels, test_data, test_labels


def load_all_data_from_zip(zipfile, data_dir, shuffle=True):
    check_and_extract_zipfile(zipfile, data_dir)
    return load_all_data(data_dir, shuffle)


def get_digits_by_label(digits, labels, query_label):
    assert digits.shape[0] == labels.shape[0]

    matching_indices = labels == query_label
    return digits[matching_indices]
