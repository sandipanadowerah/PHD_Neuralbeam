#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import sys
from code_utils.misc_utils import *
import numpy as np
import pdb


def fuse_packets(rir_step, noise, dset):
    d = []
    if dset == 'test':
        rir_start = 10001
        rir_stop = 11000
    elif dset == 'train':
        rir_start = 1
        rir_stop = 10000
    else:
        raise AttributeError('dataset should be `train` or `test`')

    for i in np.arange(rir_start, rir_stop, rir_step):
        dictry = 'results_danse_' + str(i) + '-' + str(i + rir_step - 1) + '_' + noise + '.p'
        d.append(pickle.load(open(dictry, 'rb')))
    d_ = concatenate_dicts(d)

    pickle.dump(d_, open('results_danse_' + str(rir_start) + '-' + str(rir_stop) + '_' + noise + '.p', 'wb'))


def fuse_singles(noise, dset, expe):
    root_name = 'results_' + expe +'_'
    if dset == 'test':
        rir_start = 10001
        rir_stop = 11000
    elif dset == 'train':
        rir_start = 1
        rir_stop = 10000
    else:
        raise AttributeError('dataset should be `train` or `test`')

    # Get first array and its keys
    init_dictry = root_name + str(rir_start) + '_' + noise + '.p'
    init_d = np.load(init_dictry)
    # init_dictry = root_name + str(10001) + '_Zmix-' + noise + '.npy'
    # init_d = np.load(init_dictry)[()]
    d = dict.fromkeys(init_d.keys())
    val_list = [[] for k in init_d.keys()]

    # Load all results and feed them to the whole one
    for i in np.arange(rir_start, rir_stop + 1):
        d_name = root_name + str(i) + '_' +  noise + '.p'
        a = np.load(d_name)
        # d_name = root_name + str(i) + '_Zmix-' +  noise + '.npy'
        # a = np.load(d_name)[()]
        for i, k in enumerate(a.keys()):
            val_list[i].append(a[k])
    
    for i, k in enumerate(d.keys()): 
        d[k] = val_list[i] 

    pickle.dump(d, open(root_name + str(rir_start) + '-' + str(rir_stop) + '_' + noise + '.p', 'wb'))
    # pickle.dump(d, open(root_name + str(rir_start) + '-' + str(rir_stop) + '_Zmix-' + noise + '.p', 'wb'))
        

def fuse_ites(noise, dset, expe, it_nb):
    """
    Fuse results of all iterations into a 3D array: sig x nodes x iteration
    :param noise:
    :param dset:
    :param expe:
    :param it_nb:
    :return:
    """
    root_name = 'results_' + expe + '_'
    if dset == 'test':
        rir_start = 10001
        rir_stop = 11000
    elif dset == 'train':
        rir_start = 1
        rir_stop = 10000
    else:
        raise AttributeError('dataset should be `train` or `test`')

    # Get first array and its keys
    init_dictry = root_name + str(rir_start) + '_Mix-' + noise + '_ite-1.p'
    init_d = np.load(init_dictry)
    d = dict.fromkeys(init_d.keys())
    nb_nodes = len(init_d[list(init_d)[0]])
    nb_vals = len(init_d.keys())
    val_list = [np.zeros((rir_stop - rir_start + 1, nb_nodes, it_nb)) for k in range(nb_vals)]

    # Load all results and feed them to the whole one
    for i in np.arange(rir_start, rir_stop + 1):
        for ite in range(it_nb):
            d_name = root_name + str(i) + '_Mix-' + noise + '_ite-' + str(ite + 1) + '.p'
            a = np.load(d_name)
            for j, k in enumerate(a.keys()):
                val_list[j][i - rir_start, :, ite] = a[k]

    for i, k in enumerate(d.keys()):
        d[k] = val_list[i]

    pickle.dump(d, open(root_name + str(rir_start) + '-' + str(rir_stop) + '_' + noise + '.p', 'wb'))


if __name__ == '__main__':
    method = sys.argv[1]
    noise = sys.argv[2]
    dset = sys.argv[3]

    if method == 'packets':
        rir_step = int(sys.argv[4])
        fuse_packets(rir_step, noise, dset)
    elif method == 'singles':
        expe = sys.argv[4]
        fuse_singles(noise, dset, expe)
    elif method == 'ites':
        expe = sys.argv[4]
        ite_nb = int(sys.argv[5])
        fuse_ites(noise, dset, expe, ite_nb)
    else:
        raise ValueError("fourth argument should be 'packets', 'singles' or 'ites'")

