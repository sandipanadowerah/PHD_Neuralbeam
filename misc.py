#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneous more or less useful functions
"""
import numpy as np
from keras import backend as K
from sklearn.preprocessing import scale
from code_utils.sigproc_utils import smooth_exp


def pcen(sig, s=0.025, alpha=0.98, delta=2, r=0.5):
    """
    Implementation of the per-channel energy normalization described in [1].
        :param sig:     Signal to normalize (1/2 D)
        :param s:       Smoothing constant
        :param alpha:   Exponent related to the AGC gain normalization
        :param delta:   Offset
        :param r:       Exponent
    [1] Wang YX. et al. Trainable Frontend For Robust and Far-Field Keyword Spotting. 
        In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2017.
    """
    eps = 1e-6
    nrj = abs(sig)
    nrj_smooth = smooth_exp(nrj, s) 
    sig_pcen = (nrj/((eps + nrj_smooth)**alpha) + delta)**r - delta**r 
    return sig_pcen


def normalize_seq(seq, method, mean_=0, std_=1, axis=0):
    """
    Normalize the sequence according to method given in method
    :param seq:         2D array (of mel-spectrogram). Linear values.
    :param method:      Method to normalize ('skl_scale', 'cent_norm')
                            - cent_norm : Retrieve mean of the whole dataset; divide by STD of the whole dataset 
                                          (noise of mixture required)
                            - skl_scale : use sklearn scale function to zero-center and unit-variance
    :param axis:        (int) Axis to compute mean and variance over. [0]
    :param mean_:       (float) in case of cent_norm: mean to deduct
    :param std_:        (float) in case of cent_norm: std to divide by
    :return:
    """
    if method == 'cent_norm':
        # 0-centered, unitary variance using statistics of the whole dataset
        seq -= mean_[:, np.newaxis]     # Center
        seq /= std_[:, np.newaxis]      # Unit-variance
    elif method == 'skl_scale':
        # 0-centered, unitary-variance per freq/mel band
        seq = scale(seq, axis=axis, with_mean=True, with_std=True, copy=True)
    elif method == 'norm_on_seq':
        seq -= np.mean(seq, axis=axis)[:, np.newaxis]
        seq /= np.std(seq, axis=axis)[:, np.newaxis]

    return seq


def extract_frames(x, frames_to_take='last', three_d_tensor=False):
    """
    Return only the frames of the input array corresponding to those predicted by the model.
    Shape follows keras logic: time x freq
    In case of a 3d tensor, assumed is that the wanted frame are from the first channel.
    :param x:
    :param frames_to_take:      'last'/'mid'/'all'
    :param three_d_tensor:
    :return:
    """

    # Cheat to by-pass Dimension(x) of Keras
    x_shape = [0] + [int(str(k)) for k in x.shape[1:]]
    if not three_d_tensor:
        if frames_to_take == 'last':
            return x[:, -1:, :]
        elif frames_to_take == 'mid':
            # Time dimension of x should not be None
            range_to_take = range(int(x_shape[1]/2), int(x_shape[1]/2) + 1)
            return x[:, range_to_take[0]:range_to_take[-1]+1, :]
        elif frames_to_take == 'all':
            return x
        else:
            raise ValueError('Unknown string for frames_to_take')
    else:
        if K.image_data_format() == 'channels_first':
            if frames_to_take == 'last':
                return x[:, 0, -1:, :]
            elif frames_to_take == 'mid':
                # Time dimension of x should not be None
                range_to_take = range(int(x_shape[2]/2), int(x_shape[2]/2) + 1)
                return x[:, 0, range_to_take[0]:range_to_take[-1]+1, :]
            elif frames_to_take == 'all':
                return x[:, 0, :, :]
            else:
                raise ValueError('Unknown string for frames_to_take')
        elif K.image_data_format() == 'channels_last':
            if frames_to_take == 'last':
                return x[:, -1:, :, 0]
            elif frames_to_take == 'mid':
                # Time dimension of x should not be None
                range_to_take = range(int(x_shape[1]/2), int(x_shape[1]/2) + 1)
                return x[:, range_to_take[0]:range_to_take[-1]+1, :, 0]
            elif frames_to_take == 'all':
                return x[:, :, :, 0]
            else:
                raise ValueError('Unknown string for frames_to_take')


def extract_frames_outshape(input_shape, frames_to_take=-1, three_d_tensor=False):
    output_shape = list(input_shape)
    channels_order = ['channels_last', 'channels_first']
    if frames_to_take in ['last', 'mid']:
        nb_frames = 1
    elif frames_to_take == 'all':
        if not three_d_tensor:
            nb_frames = output_shape[1]
        else:
            nb_frames = output_shape[channels_order.index(K.image_data_format())] + 2
    else:
        raise ValueError('Unknown string for frames_to_take')

    if not three_d_tensor:
        if frames_to_take == 'all':
            nb_frames = output_shape[1]
        output_shape[1] = nb_frames
    else:
        if K.image_data_format() == 'channels_first':
            output_shape[3] = nb_frames
        elif K.image_data_format() == 'channels_last':
            output_shape[2] = nb_frames
    return tuple(output_shape)
