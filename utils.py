import datetime
import random
import sys
import time

import numpy as np

import torch
from sklearn import preprocessing
from torch.autograd import Variable


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)

            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def create_lable_func(min_val, max_val, n_bins):
    ''' genrate function to encode continus values to n classes
        return mappingf funciton to lable or to one hot lable
        usage:
        labl, hot_l = creat_lable_func(0, 10, 5)
        x_fit = np.linspace(0, 10, 11)
        print("y leables", labl(x_fit))
        print("yhot leables",  hot_l(x_fit))
    '''
    x_fit = np.linspace(min_val, max_val, 5000)
    bins = np.linspace(min_val, max_val, n_bins)
    x_fit = np.digitize(x_fit, bins)
    le = preprocessing.LabelEncoder()
    le.fit(x_fit)
    le_one_hot = preprocessing.LabelBinarizer()
    assert len(le.classes_) == n_bins
    le_one_hot.fit(le.classes_)
    y = le.transform(x_fit)
    yhot = le_one_hot.transform(x_fit)

    def _enc_lables(data):
        fit = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        return le.transform(np.digitize(fit, bins))

    def _enc_lables_hot(data):
        fit = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        return le_one_hot.transform(np.digitize(fit, bins))
    return _enc_lables, _enc_lables_hot
