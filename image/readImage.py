# -*- coding: utf-8 -*-


import json
import os
# import urllib2
import httplib2

import cv2
import matplotlib.pyplot as plt

from io_util.image import loadRGB, saveRGB

_root_dir = os.path.dirname(__file__)


## Data directory for the given data_name.
def dataDir(data_name):
    data_dir = os.path.join(_root_dir, data_name)
    # print data_name
    # print _root_dir
    return data_dir


## Data file path list for the given data_name.
def dataFiles(data_name):
    data_dir = dataDir(data_name)
    data_files = []
    for data_name in os.listdir(data_dir):
        data_file = os.path.join(data_dir, data_name)
        if ".png" in data_name or ".jpg" in data_name:
            data_files.append(data_file)
        else:
            os.remove(data_file)
    return data_files


## Data file path for the given data_name and data_id.
def dataFile(data_name, data_id):
    data_files = dataFiles(data_name)

    if data_id >= len(data_files):
        return None

    data_file = data_files[data_id]
    return data_file


def loadData(data_name, data_id):
    data_file = dataFile(data_name, data_id)

    if data_file is None:
        return None

    return loadRGB(data_file)
