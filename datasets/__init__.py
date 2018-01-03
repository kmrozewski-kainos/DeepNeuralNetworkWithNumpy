#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os



class Dataset(object):
    
    def load_data(self, filepath):
        if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
            dataset = h5py.File(filepath, "r")
            return dataset
        else:
            print 'File %s doesnt exists or no priviledges to open it' % filepath


class Train(Dataset):
    
    def __init__(self):
        self.filepath = 'datasets/train_catvnoncat.h5'
        dataset = self.load_data(self.filepath)
        
        x = np.array(dataset['train_set_x'][:])
        y = np.array(dataset['train_set_y'][:])
        
        self.dataset = {'x': x, 'y': y}


class Test(Dataset):
    
    def __init__(self):
        self.filepath = 'datasets/test_catvnoncat.h5'
        dataset = self.load_data(self.filepath)
        
        x = np.array(dataset["test_set_x"][:])
        y = np.array(dataset["test_set_y"][:])
        classes = np.array(dataset["list_classes"][:])
        
        self.dataset = {'x': x, 'y': y, 'classes': classes}


test = Test()
test.dataset.keys()