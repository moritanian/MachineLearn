# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

def import_cifar():
    X_train = None
    Y_train = []

    for i in range(1,6):
        data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        Y_train += data_dic['labels']

    test_data_dic = unpickle("cifar-10-batches-py/test_batch")
    X_test = test_data_dic['data']
    X_test = X_test.reshape(len(X_test),3,32,32)
    Y_test = np.array(test_data_dic['labels'])
    X_train = X_train.reshape((len(X_train),3, 32, 32))
    Y_train = np.array(Y_train)

    # print (y_train)
    # print (y_train.size)
    return {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}

def get_image_from_cifar(arr):
    width = len(arr[0][0])
    height = len(arr[0]) 
    #print width
    #print height
    img_arr = np.zeros([height, width, 3])
    for x in range(width):
        for y in range(height):
            img_arr[y][x][0] = int(arr[0][y][x]) 
            img_arr[y][x][1] = int(arr[1][y][x]) 
            img_arr[y][x][2] = int(arr[2][y][x])
    return  Image.fromarray(np.uint8(img_arr))

def test():
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    num = 7
    data = import_cifar()
    arr = get_image_from_cifar(data['X_train'][num])
    arr.show()
    print names[data['Y_train'][num]]
    #print arr.shape
  