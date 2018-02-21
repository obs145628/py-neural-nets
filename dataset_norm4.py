import numpy as np
import pickle

TRAIN_TEST_FILE = './norm_100k_10k.bin'
MINI_TRAIN_TEST_FILE = './norm_1k_100.bin'

def vec_to_num(x):
    return np.argmax(x)

def num_to_vec(x):
    if x >= 1:
        return np.array([0, 1])
    else:
        return np.array([1, 0])

def gen_train_test(train_len, test_len):
    total_len = train_len + test_len

    X = np.zeros((total_len, 4))
    y = np.zeros((total_len, 2))

    for i in range(0, total_len):
        X[i] = np.random.rand(4)
        y[i] = num_to_vec(np.linalg.norm(X[i]))


    X_train = X[:train_len]
    y_train = y[:train_len]
    X_test = X[test_len:]
    y_test = y[test_len:]

    return X_train, y_train, X_test, y_test


def build_data_train_test(path, train_len, test_len):
    print('Building dataset...')
    data = gen_train_test(train_len, test_len)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print('Dataset built')
    return data

def load_data_train_test(path, train_len, test_len):
    try:
        f = open(path, 'rb')
    except IOError:
        return build_data_train_test(path, train_len, test_len)
    with f:
        return pickle.load(f)

def load_norm4():
    return load_data_train_test(TRAIN_TEST_FILE, 100000, 10000)

def load_mini_norm4():
    return load_data_train_test(MINI_TRAIN_TEST_FILE, 1000, 100)

def output_test(y_hat, y):
    return vec_to_num(y_hat) == vec_to_num(y)
