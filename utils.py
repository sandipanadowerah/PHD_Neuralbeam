import numpy as np
np.random.seed(10)
import pyroomacoustics as pra
import soundfile as sf
import sys
from code_utils.math_utils import cart2pol, pol2cart, floor_to_multiple, db2lin
from code_utils.sigproc_utils import vad_oracle_batch
from code_utils.metrics import snr, seg_snr, reverb_ratios
from code_utils.db_utils import *
from code_utils.mask_utils import wiener_mask
import math
import glob
import time
import matplotlib.pyplot as plt
import ipdb
from hyperparameters import *


def rir_id_exists(path_out_data, rir_id):
    path = os.path.join(path_out_data+'/dry_noise/', str(rir_id) + '.wav')
    #print(path)
    return os.path.exists(path)


def pad_to_maxlen(x, y):
    diff = len(x) - len(y)
    if diff > 0 :
        y = np.concatenate([y, np.zeros(diff)])	
    elif diff < 0 :
        x = np.concatenate([x, np.zeros(abs(diff))])

    return x, y

def get_room_configuration():
    # Geometric properties
    length = len_min + (len_max - len_min)*np.random.rand()
    width = wid_min + (wid_max - wid_min)*np.random.rand()
    height = hei_min + (hei_max - hei_min)*np.random.rand()
    vol = length * width * height
    sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

    # Acoustic properties
    beta = beta_min + (beta_max - beta_min) * np.random.rand()
    alpha = 1 - np.exp((0.017 * beta - 0.1611)*vol/(beta * sur))

    return length, width, height, alpha, beta

def get_room_configuration_beta(beta):
    # Geometric properties
    length = len_min + (len_max - len_min)*np.random.rand()
    width = wid_min + (wid_max - wid_min)*np.random.rand()
    height = hei_min + (hei_max - hei_min)*np.random.rand()
    vol = length * width * height
    sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

    # Acoustic properties
    #beta = beta_min + (beta_max - beta_min) * np.random.rand()
    alpha = 1 - np.exp((0.017 * beta - 0.1611)*vol/(beta * sur))

    return length, width, height, alpha, beta

def get_room_configuration_multiple_beta(beta1, beta2, beta3):
    # Geometric properties
    length = len_min + (len_max - len_min)*np.random.rand()
    width = wid_min + (wid_max - wid_min)*np.random.rand()
    height = hei_min + (hei_max - hei_min)*np.random.rand()
    vol = length * width * height
    sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

    # Acoustic properties
    #beta = beta_min + (beta_max - beta_min) * np.random.rand()
    alpha1 = 1 - np.exp((0.017 * beta1 - 0.1611)*vol/(beta1 * sur))
    alpha2 = 1 - np.exp((0.017 * beta2 - 0.1611)*vol/(beta2 * sur))
    alpha3 = 1 - np.exp((0.017 * beta3 - 0.1611)*vol/(beta3 * sur))

    return length, width, height, alpha1, alpha2, alpha3


def get_array_positions(length, width):
    """

    :param length:
    :param width:
    :param phi:
    :return:        n1 - first node centre position
                    n2 - Second node centre position
                    o  - Array centre position
    """
    phi = np.pi*np.random.rand()                # random orientation in the room (in rad). Symmetric so up to pi only

    # Position of the microphone in the referential of the node
    x_mics_local = [d_mic * np.cos(phi),            # x-positions
                    d_mic * np.sin(phi),            # First mic is mic
                    -d_mic * np.cos(phi),           # in extension of array;
                    -d_mic * np.sin(phi)]
    x_mic_max = np.max(x_mics_local)

    y_mics_local = [d_mic * np.sin(phi),            # y-positions
                    -d_mic * np.cos(phi),
                    -d_mic * np.sin(phi),
                    d_mic * np.cos(phi)]
    y_mic_max = np.max(y_mics_local)

    # Position of the array *centre*
    o_x_min = d_wal + d/2*abs(np.cos(phi)) + x_mic_max          # Smallest allowed x-position (distance to x=0 axis)
    o_x_max = length - o_x_min
    o_x = o_x_min + (o_x_max - o_x_min) * np.random.rand()      # Random x-position in allowed area

    o_y_min = d_wal + d/2 * np.sin(phi) + y_mic_max
    o_y_max = width - o_y_min
    o_y = o_y_min + (o_y_max - o_y_min) * np.random.rand()      # Random x-position in allowed area

    # Nodes centre positions
    n1_x = o_x + d/2 * np.cos(phi)
    n1_y = o_y + d/2 * np.sin(phi)
    n2_x = o_x - d/2 * np.cos(phi)
    n2_y = o_y - d/2 * np.sin(phi)

    return [[n1_x, n1_y, z_cst], [n2_x, n2_y, z_cst]], [o_x, o_y, z_cst], phi


def get_noise_type_list(path_robovox, case):
    noise_type_list = {}
   
    if case == 'train':
        for line in open(os.path.join(path_robovox, 'training.csv'), 'r'):
            noise_type_list[line.strip().split(',')[3]] = line.strip().split(',')[1]
    else:
        for line in open(os.path.join(path_robovox, 'test.csv'), 'r'):
            noise_type_list[line.strip().split(',')[3]] = line.strip().split(',')[1]

    return noise_type_list


def get_array_positions_phi(length, width, phi):
    """

    :param length:
    :param width:
    :param phi:
    :return:        n1 - first node centre position
                    n2 - Second node centre position
                    o  - Array centre position
    """
    #phi = np.pi*np.random.rand()                # random orientation in the room (in rad). Symmetric so up to pi only

    # Position of the microphone in the referential of the node
    x_mics_local = [d_mic * np.cos(phi),            # x-positions
                    d_mic * np.sin(phi),            # First mic is mic
                    -d_mic * np.cos(phi),           # in extension of array;
                    -d_mic * np.sin(phi)]
    x_mic_max = np.max(x_mics_local)

    y_mics_local = [d_mic * np.sin(phi),            # y-positions
                    -d_mic * np.cos(phi),
                    -d_mic * np.sin(phi),
                    d_mic * np.cos(phi)]
    y_mic_max = np.max(y_mics_local)

    # Position of the array *centre*
    o_x_min = d_wal + d/2*abs(np.cos(phi)) + x_mic_max          # Smallest allowed x-position (distance to x=0 axis)
    o_x_max = length - o_x_min
    o_x = o_x_min + (o_x_max - o_x_min) * np.random.rand()      # Random x-position in allowed area

    o_y_min = d_wal + d/2 * np.sin(phi) + y_mic_max
    o_y_max = width - o_y_min
    o_y = o_y_min + (o_y_max - o_y_min) * np.random.rand()      # Random x-position in allowed area

    # Nodes centre positions
    n1_x = o_x + d/2 * np.cos(phi)
    n1_y = o_y + d/2 * np.sin(phi)
    n2_x = o_x - d/2 * np.cos(phi)
    n2_y = o_y - d/2 * np.sin(phi)

    return [[n1_x, n1_y, z_cst], [n2_x, n2_y, z_cst]], [o_x, o_y, z_cst], phi


def get_random_mics_positions(length, width):
    """
    Return the (x, y, z) coordinates of two microphones randomly placed in the room, at least distant of d_wal from the
    walls and d_rnd_mic from each other
    :param length:
    :param width:
    :return:
    """
    m1_x = d_wal + (length - 2 * d_wal) * np.random.rand()
    m1_y = d_wal + (width - 2 * d_wal) * np.random.rand()

    m2_x = d_wal + (length - 2 * d_wal) * np.random.rand()
    m2_y = d_wal + (width - 2 * d_wal) * np.random.rand()

    while np.sqrt((m1_x - m2_x)**2 + (m1_y - m2_y)**2) < d_rnd_mics:
        m2_x = d_wal + (length - 2 * d_wal) * np.random.rand()
        m2_y = d_wal + (width - 2 * d_wal) * np.random.rand()

    return [m1_x, m1_y, z_cst], [m2_x, m2_y, z_cst]

def get_theta(mic_xyz, source_position):
    
    dy = mic_xyz[1] - source_position[1]
    dx = mic_xyz[0] - source_position[0]

    phi = np.rad2deg(np.arctan(dy/dx))

    return phi

def get_source_positions_sfn(length, width, nodes_center, d_to_nodes=d_sou):
    """

    :param length:
    :param width:
    :param nodes_center:        Nodes central position (avoid source too close to nodes)
    :param d_to_nodes:          Distance to nodes (avoid source too close to nodes)
    :return:
        - Sources positions (x, y, z)
        - a counter: if equal to 100, no configuration was found and new input arguments should be given
    """
    ss = [[], []]
    ss_angle = 180
    
    min_angle_tolerence, max_angle_tolerence = -1.0,1.0
    cnt_alpha = 0
    for i in range(2):
        if cnt_alpha < 1000:     # Check (for i=1) that previous source is OK
            cnt_alpha = 0       # Reset to 0 after first source is found
            p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
            p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
            
            if i == 1:
                while (np.sqrt((nodes_center[0][0] - p_x) ** 2 + (nodes_center[0][1] - p_y) ** 2) < d_to_nodes
                      or np.sqrt((nodes_center[1][0] - p_x) ** 2 + (nodes_center[1][1] - p_y) ** 2) < d_to_nodes) \
                      and cnt_alpha < 1000:
                    p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
                    p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
                    cnt_alpha += 1
                ss[i] = [p_x, p_y, z_cst]
            elif i == 0:
                angle = get_theta([nodes_center[0][0], nodes_center[0][1]], [p_x, p_y])
                while (np.sqrt((nodes_center[0][0] - p_x) ** 2 + (nodes_center[0][1] - p_y) ** 2) < d_to_nodes
                      or np.sqrt((nodes_center[1][0] - p_x) ** 2 + (nodes_center[1][1] - p_y) ** 2) < d_to_nodes) \
                      and cnt_alpha < 1000 or (min_angle_tolerence > angle or angle > max_angle_tolerence):
                    p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
                    p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
                    angle = get_theta([nodes_center[0][0], nodes_center[0][1]], [p_x, p_y])
                    
                    cnt_alpha += 1
                ss[i] = [p_x, p_y, z_cst]
                ss_angle = angle
                #print(ss_angle, 'facing angle')
        else:
            return ss, cnt_alpha, ss_angle
        
        
        
    return ss, cnt_alpha, angle



def get_source_positions(length, width, nodes_center, d_to_nodes=d_sou):
    """

    :param length:
    :param width:
    :param nodes_center:        Nodes central position (avoid source too close to nodes)
    :param d_to_nodes:          Distance to nodes (avoid source too close to nodes)
    :return:
        - Sources positions (x, y, z)
        - a counter: if equal to 100, no configuration was found and new input arguments should be given
    """
    ss = [[], []]

    cnt_alpha = 0
    for i in range(2):
        if cnt_alpha < 100:     # Check (for i=1) that previous source is OK
            cnt_alpha = 0       # Reset to 0 after first source is found
            p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
            p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
            while (np.sqrt((nodes_center[0][0] - p_x) ** 2 + (nodes_center[0][1] - p_y) ** 2) < d_to_nodes
                  or np.sqrt((nodes_center[1][0] - p_x) ** 2 + (nodes_center[1][1] - p_y) ** 2) < d_to_nodes) \
                  and cnt_alpha < 100:
                p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
                p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
                cnt_alpha += 1
            ss[i] = [p_x, p_y, z_cst]
        else:
            return ss, cnt_alpha
    return ss, cnt_alpha


def get_target_segments(target_file, min_duration, max_duration):
    """
    Return source signals (one noise, one target)
    :param target_file:     name of a .wav/.flac file
    :return:                If target_file is long enough, the reshaped signal; if too short, None
    """
    signal, fs = sf.read(target_file)
    signal = signal[:, np.newaxis]
    sig_duration = len(signal) / fs
    #print(sig_duration)

    if sig_duration < min_duration:
        ssignal = -1
        vsignal = -1
    else:
        # If signal too long, reshape it into several segments
        if sig_duration > max_duration:
            signal = np.reshape(signal[:floor_to_multiple(sig_duration * fs, max_duration * fs)],
                                (max_duration * fs, np.int(np.floor(sig_duration / max_duration))),
                                order='F')
        # Add one second of silence at the beginning
        nb_seg = signal.shape[1]
        ssignal = np.zeros((signal.shape[0] + fs, nb_seg))
        vsignal = np.zeros(ssignal.shape)
        for i_seg in np.arange(nb_seg):
            # VAD
            vad_signal = vad_oracle_batch(signal[:, i_seg], thr=0.001)
            # Normalize the segment
            signal[:, i_seg] *= np.sqrt(var_max / np.var(signal[vad_signal == 1, i_seg]))
            ssignal[:, i_seg] = np.concatenate((np.zeros(fs), signal[:, i_seg]))
            vsignal[:, i_seg] = np.concatenate((np.zeros(fs), vad_signal))

    return ssignal, vsignal, fs


def get_noise_segment(noise_type, robovox_list, duration):
    #print(n_type)#TODO
    #print('get_noise_segment', len(robovox_list))

    n, fs, n_file, n_file_start = read_random_part(robovox_list, duration)
    noise_vad = None
   
    return n, n_file, n_file_start, noise_vad, fs


def pad_to_length(signal, length):
    """
    Pad with 0 a signal (1-D) to desired length
    :param signal:
    :param length:
    :return:
    """
    samples_to_pad = np.max((int(length - len(signal)), 0))
    padded_signal = np.pad(signal, (0, samples_to_pad), 'constant', constant_values=0)
    return padded_signal



import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn 
import numpy as np
import librosa as lb
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import soundfile as sf
import fnmatch
import collections
import random
import sys
import json
import struct
import wave
import librosa
import numpy as np
import numpy as np
from code_utils.math_utils import db2lin
from code_utils.db_utils import stack_talkers
import librosa as lb
import sys

eps = sys.float_info.epsilon
import mir_eval
import numpy as np
import os
import soundfile as sf
import scipy
import sys
import json

import timeit

import pickle
import os
import numpy as np

from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.preprocessing.data import _handle_zeros_in_scale

import scipy.stats

from beamformer import delaysum as ds
from beamformer import util


STFT_MIN = 1e-6
STFT_MAX = 1e3


# In[27]:

def eval_mir_i(estimated_sources, dry_reference_sources, reference_sources=None):
    sdr_, sir_, sar_, _ = mir_eval.separation.bss_eval_sources(np.transpose(dry_reference_sources), np.transpose(estimated_sources), compute_permutation=False)
    #sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(np.transpose(reference_sources), np.transpose(estimated_sources), compute_permutation=False)
    return (max(sdr_), max(sir_), max(sar_))

def mir_eval_source_image_models(s, n, s_dry, n_dry, y, y_out):
    y_unet_mwf = y_out[0]
    y_unet_masking = y_out[1]
    y_crnn_mwf = y_out[2]
    y_crnn_masking = y_out[3]
    y_irm = y_out[4]
    
    min_len = min(s.shape[0], s_dry.shape[0])
    dry_reference_sources = np.hstack((s_dry[:min_len, :], 
                                       n_dry[:min_len, :]))
    #reference_sources = np.hstack((s[:min_len, :], 
    #                                   n[:min_len, :]))
    estimated_sources_unet_mwf = np.hstack((y_unet_mwf[:min_len, :], 
                                            y[:min_len, :] - y_unet_mwf[:min_len, :]))
    estimated_sources_unet_masking = np.hstack((y_unet_masking[:min_len, :], 
                                                y[:min_len, :] - y_unet_masking[:min_len, :]))
    estimated_sources_crnn_mwf = np.hstack((y_crnn_mwf[:min_len, :], 
                                            y[:min_len, :] - y_crnn_mwf[:min_len, :]))
    estimated_sources_crnn_masking = np.hstack((y_crnn_masking[:min_len, :], 
                                                y[:min_len, :] - y_crnn_masking[:min_len, :]))
    estimated_sources_irm = np.hstack((y_irm[:min_len, :], 
                                       y[:min_len, :] - y_irm[:min_len, :]))
    
    e_unet_mwf = eval_mir_i(estimated_sources_unet_mwf, dry_reference_sources)
    e_unet_ = eval_mir_i(estimated_sources_unet_masking, dry_reference_sources)
    e_crnn_mwf = eval_mir_i(estimated_sources_crnn_mwf, dry_reference_sources)
    e_crnn_ = eval_mir_i(estimated_sources_crnn_masking, dry_reference_sources) 
    e_irm = eval_mir_i(estimated_sources_irm, dry_reference_sources)
    
    return e_unet_mwf, e_unet_, e_crnn_mwf, e_crnn_, e_irm


def eval_mir(estimated_sources, dry_reference_sources):
    sdr_, sir_, sar_, _ = mir_eval.separation.bss_eval_sources(np.transpose(dry_reference_sources), 
                                                                  np.transpose(estimated_sources), compute_permutation=False)

    sdr_i, isr, sir_i, sar_i, _ = mir_eval.separation.bss_eval_images(np.transpose(dry_reference_sources), 
                                                                  np.transpose(estimated_sources), compute_permutation=False)
   
    return (sdr_[0], sir_[0], sar_[0], sdr_i[0], sir_i[0], sar_i[0])

FS = 16000

def mir_eval_source_image(s, n, s_dry, n_dry, y, y_out1):

    min_len = min(s.shape[0], s_dry.shape[0])     
    
    dry_reference_sources = np.hstack((s_dry[:min_len, :], n_dry[:min_len, :]))
    reference_sources = np.hstack((s[:min_len, :], n[:min_len, :]))

    estimated_sources_out1 = np.hstack((y_out1[:min_len, :], y[:min_len, :] - y_out1[:min_len, :]))

    eval_out1 = eval_mir(estimated_sources_out1, dry_reference_sources)   
    
    return eval_out1

def mir_eval_source_image_custom(s, n, s_dry, n_dry, y, y_out1, y_out2, y_out3, y_out4):

    min_len = min(s.shape[0], s_dry.shape[0])     
    
    dry_reference_sources = np.hstack((s_dry[:min_len, :], n_dry[:min_len, :]))
    reference_sources = np.hstack((s[:min_len, :], n[:min_len, :]))

    estimated_sources_out1 = np.hstack((y_out1[:min_len, :], y[:min_len, :] - y_out1[:min_len, :]))

    eval_out1 = eval_mir(estimated_sources_out1, dry_reference_sources)

    estimated_sources_out2 = np.hstack((y_out2[:min_len, :], y[:min_len, :] - y_out2[:min_len, :]))

    eval_out2 = eval_mir(estimated_sources_out2, dry_reference_sources)

    estimated_sources_out3 = np.hstack((y_out3[:min_len, :], y[:min_len, :] - y_out3[:min_len, :]))

    eval_out3 = eval_mir(estimated_sources_out3, dry_reference_sources)

    estimated_sources_out4 = np.hstack((y_out4[:min_len, :], y[:min_len, :] - y_out4[:min_len, :]))

    eval_out4 = eval_mir(estimated_sources_out4, dry_reference_sources)
    
    return eval_out1,eval_out2,eval_out3,eval_out4

def mir_eval_source_image_custom2(s, n, s_dry, n_dry, y, y_out1, y_out2, y_out3, y_out4, y_out5, y_out6, y_out7, y_out8, y_out9, y_out10, y_out11, y_out12):

    min_len = min(s.shape[0], s_dry.shape[0])     
    
    dry_reference_sources = np.hstack((s_dry[:min_len, :], n_dry[:min_len, :]))
    reference_sources = np.hstack((s[:min_len, :], n[:min_len, :]))

    estimated_sources_out1 = np.hstack((y_out1[:min_len, :], y[:min_len, :] - y_out1[:min_len, :]))

    eval_out1 = eval_mir(estimated_sources_out1, dry_reference_sources)

    estimated_sources_out2 = np.hstack((y_out2[:min_len, :], y[:min_len, :] - y_out2[:min_len, :]))

    eval_out2 = eval_mir(estimated_sources_out2, dry_reference_sources)

    estimated_sources_out3 = np.hstack((y_out3[:min_len, :], y[:min_len, :] - y_out3[:min_len, :]))

    eval_out3 = eval_mir(estimated_sources_out3, dry_reference_sources)

    estimated_sources_out4 = np.hstack((y_out4[:min_len, :], y[:min_len, :] - y_out4[:min_len, :]))

    eval_out4 = eval_mir(estimated_sources_out4, dry_reference_sources)

    estimated_sources_out5 = np.hstack((y_out5[:min_len, :], y[:min_len, :] - y_out5[:min_len, :]))

    eval_out5 = eval_mir(estimated_sources_out5, dry_reference_sources)

    estimated_sources_out6 = np.hstack((y_out6[:min_len, :], y[:min_len, :] - y_out6[:min_len, :]))

    eval_out6 = eval_mir(estimated_sources_out6, dry_reference_sources)

    estimated_sources_out7 = np.hstack((y_out7[:min_len, :], y[:min_len, :] - y_out7[:min_len, :]))

    eval_out7 = eval_mir(estimated_sources_out7, dry_reference_sources)

    estimated_sources_out8 = np.hstack((y_out8[:min_len, :], y[:min_len, :] - y_out8[:min_len, :]))

    eval_out8 = eval_mir(estimated_sources_out8, dry_reference_sources)

    estimated_sources_out9 = np.hstack((y_out9[:min_len, :], y[:min_len, :] - y_out9[:min_len, :]))

    eval_out9 = eval_mir(estimated_sources_out9, dry_reference_sources)

    estimated_sources_out10 = np.hstack((y_out10[:min_len, :], y[:min_len, :] - y_out10[:min_len, :]))

    eval_out10 = eval_mir(estimated_sources_out10, dry_reference_sources)

    estimated_sources_out11 = np.hstack((y_out11[:min_len, :], y[:min_len, :] - y_out11[:min_len, :]))

    eval_out11 = eval_mir(estimated_sources_out11, dry_reference_sources)

    estimated_sources_out12 = np.hstack((y_out12[:min_len, :], y[:min_len, :] - y_out12[:min_len, :]))

    eval_out12 = eval_mir(estimated_sources_out12, dry_reference_sources)

    
    return eval_out1,eval_out2,eval_out3, eval_out4, eval_out5, eval_out6, eval_out7, eval_out8, eval_out9, eval_out10, eval_out11, eval_out12

def mir_eval_source_image_custom3(s, n, s_dry, n_dry, y, y_out1, y_out2, y_out3, y_out4, y_out5, y_out6, y_out7, y_out8):

    min_len = min(s.shape[0], s_dry.shape[0])     
    
    dry_reference_sources = np.hstack((s_dry[:min_len, :], n_dry[:min_len, :]))
    reference_sources = np.hstack((s[:min_len, :], n[:min_len, :]))

    estimated_sources_out1 = np.hstack((y_out1[:min_len, :], y[:min_len, :] - y_out1[:min_len, :]))

    eval_out1 = eval_mir(estimated_sources_out1, dry_reference_sources)

    estimated_sources_out2 = np.hstack((y_out2[:min_len, :], y[:min_len, :] - y_out2[:min_len, :]))

    eval_out2 = eval_mir(estimated_sources_out2, dry_reference_sources)

    estimated_sources_out3 = np.hstack((y_out3[:min_len, :], y[:min_len, :] - y_out3[:min_len, :]))

    eval_out3 = eval_mir(estimated_sources_out3, dry_reference_sources)

    estimated_sources_out4 = np.hstack((y_out4[:min_len, :], y[:min_len, :] - y_out4[:min_len, :]))

    eval_out4 = eval_mir(estimated_sources_out4, dry_reference_sources)

    estimated_sources_out5 = np.hstack((y_out5[:min_len, :], y[:min_len, :] - y_out5[:min_len, :]))

    eval_out5 = eval_mir(estimated_sources_out5, dry_reference_sources)

    estimated_sources_out6 = np.hstack((y_out6[:min_len, :], y[:min_len, :] - y_out6[:min_len, :]))

    eval_out6 = eval_mir(estimated_sources_out6, dry_reference_sources)

    estimated_sources_out7 = np.hstack((y_out7[:min_len, :], y[:min_len, :] - y_out7[:min_len, :]))

    eval_out7 = eval_mir(estimated_sources_out7, dry_reference_sources)

    estimated_sources_out8 = np.hstack((y_out8[:min_len, :], y[:min_len, :] - y_out8[:min_len, :]))

    eval_out8 = eval_mir(estimated_sources_out8, dry_reference_sources)

    
    return eval_out1,eval_out2,eval_out3, eval_out4, eval_out5, eval_out6, eval_out7, eval_out8

    


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h



def save_wav(y, sr, wav_dir, rir_id, nch=1):
    nch = y.shape[1]
    for i in range(1,nch+1):
        #print(i)
        str_ = '_output_wav_Ch-%d.wav'%i
        outwavpath = os.path.join(wav_dir, str(rir_id)+str_)
        sf.write(outwavpath, y[:,i-1], samplerate=sr)

def ideal_ratio_mask(s_stft, n_stft):
    mask = np.square(np.abs(s_stft))/(np.square(np.abs(s_stft)) + np.square(np.abs(n_stft)))
    return mask

def get_data(index, y_dir, s_dir, n_dir, s_dir_dry, n_dir_dry):
    y, sr = load_wav_data(index, y_dir, stype='y') 
    s, sr = load_wav_data(index, s_dir, stype='s')
    n, sr = load_wav_data(index, n_dir, stype='n')
    s_dry, sr = load_wav_data(index, s_dir_dry, stype='d')
    n_dry, sr = load_wav_data(index, n_dir_dry, stype='d')

    y_stft = stft_nchannel_wave(y)
    s_stft = stft_nchannel_wave(s)
    n_stft = stft_nchannel_wave(n)

    mask = ideal_ratio_mask(s_stft, n_stft)

    return y_stft, y, s, n, s_dry, n_dry, mask, sr

def get_theta_phi(mic_xyz, source_position):
        phi = []
        for (x,y,z) in mic_xyz:
            #print(x,y,z)
            dy = y - source_position[1]
            dx = x - source_position[0]

            phi.append(np.rad2deg(np.arctan(dy/dx)))

        return phi

def delay_and_sum_beamforming_wav(y_audio, phi, win_len = 512, hop_len = 256, sr=16000):
    SAMPLING_FREQUENCY = sr
    FFT_LENGTH = win_len
    FFT_SHIFT = hop_len
    MIC_ANGLE_VECTOR = np.array([0, 120, 360])
    LOOK_DIRECTION = phi
    MIC_DIAMETER = 0.1

    complex_spectrum, _ = util.get_3dim_spectrum_from_data(y_audio, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
    complex_spectrum =  np.clip(complex_spectrum, STFT_MIN, STFT_MAX)

    ds_beamformer = ds.delaysum(MIC_ANGLE_VECTOR, MIC_DIAMETER, 
                                                      sampling_frequency=SAMPLING_FREQUENCY, 
                                                      fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

    beamformer = ds_beamformer.get_sterring_vector(LOOK_DIRECTION)

    enhanced_speech = ds_beamformer.apply_beamformer(beamformer, complex_spectrum)

    norm_wav = enhanced_speech / np.max(np.abs(enhanced_speech))

    len_diff = y_audio.shape[0] - norm_wav.shape[0]

    if len_diff < 0:   
        norm_wav = norm_wav[:y_audio.shape[0],]
    elif len_diff > 0:    
        norm_wav = np.concatenate((norm_wav, np.zeros(abs(len_diff))))


    return norm_wav 

def delay_and_sum_beamforming(y_audio, phi, win_len = 512, hop_len = 256, sr=16000):
    SAMPLING_FREQUENCY = sr
    FFT_LENGTH = win_len
    FFT_SHIFT = hop_len
    MIC_ANGLE_VECTOR = np.array([0, 120, 360])
    LOOK_DIRECTION = phi
    MIC_DIAMETER = 0.1

    complex_spectrum, _ = util.get_3dim_spectrum_from_data(y_audio, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
    complex_spectrum =  np.clip(complex_spectrum, STFT_MIN, STFT_MAX)

    ds_beamformer = ds.delaysum(MIC_ANGLE_VECTOR, MIC_DIAMETER, 
                                                      sampling_frequency=SAMPLING_FREQUENCY, 
                                                      fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

    beamformer = ds_beamformer.get_sterring_vector(LOOK_DIRECTION)

    enhanced_speech = ds_beamformer.apply_beamformer(beamformer, complex_spectrum)

    norm_wav = enhanced_speech / np.max(np.abs(enhanced_speech))

    len_diff = y_audio.shape[0] - norm_wav.shape[0]

    if len_diff < 0:   
        norm_wav = norm_wav[:y_audio.shape[0],]
    elif len_diff > 0:    
        norm_wav = np.concatenate((norm_wav, np.zeros(abs(len_diff))))

    norm_stft = stft_nchannel_wave(norm_wav, n_channel=1)

    return norm_stft, norm_wav
    
def get_data_beamforming(index, y_dir, s_dir, n_dir, s_dir_dry, n_dir_dry, rir_dir):
    y, sr = load_wav_data(index, y_dir, stype='y') 
    s, sr = load_wav_data(index, s_dir, stype='s')
    n, sr = load_wav_data(index, n_dir, stype='n')
    s_dry, sr = load_wav_data(index, s_dir_dry, stype='d')
    n_dry, sr = load_wav_data(index, n_dir_dry, stype='d')

    y_stft = stft_nchannel_wave(y)
    s_stft = stft_nchannel_wave(s)
    n_stft = stft_nchannel_wave(n)

    mask = ideal_ratio_mask(abs(s_stft), abs(n_stft))
    rir_info = np.load(os.path.join(rir_dir, str(index)+'_info.npy'), allow_pickle=True).any()
    theta = get_theta_phi(rir_info['mics_xyz'], rir_info['sous_xyz'][0])
    #print(theta)
    ds_stft = delay_and_sum_beamforming(y, theta)
                
    #np.save(os.path.join(self.stft_mask_dir, str(index)+'.npy'), {'mask':mask, 'ds_stft':ds_stft}, allow_pickle=True)
       
    y_stft_ = np.concatenate((abs(y_stft), np.expand_dims(abs(ds_stft), axis=2)), axis=2)
    #y_stft = torch.from_numpy(abs(y_stft))
    #s_stft = torch.from_numpy(abs(s_stft))
    #mask = torch.from_numpy(mask)
        
        
    return y_stft, y_stft_, y, s, n, s_dry, n_dry, mask, sr
    



def get_fileid_list(array_file):
    list_ = []
    for i in open(array_file, 'r'):
        list_.append(int(i.strip()))
    return list_




def stft_nchannel_wave(y, win_len=512, win_hop=256, n_channel=3, center=False):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((y.shape[0] - win_len) / win_hop))

    y_stft = np.zeros((n_freq, n_frames, n_channel), 'complex')

    if n_channel == 1:
        y_stft = librosa.core.stft(np.ascontiguousarray(y), n_fft=win_len, hop_length=win_hop, center=False)
    else:
        for i_ch in range(n_channel):
            y_stft[:, :, i_ch] = librosa.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)


    return np.clip(abs(y_stft), STFT_MIN, STFT_MAX)


def load_wav_data(rir_id, wav_dirpath, nch=3, stype = 'y'):
    wav_data = []
    for i in range(1,nch+1):
        if stype == 's':
            str_ = str(rir_id) +'_target_Ch-%d.wav'%i
        elif stype == 'n':
            str_ = str(rir_id) + '_robovox_Ch-%d.wav'%i
        elif stype == 'y':
            str_ = str(rir_id) + '_Mix-robovox_Ch-%d.wav'%i
        elif stype == 'd':
            str_ = str(rir_id) + '.wav'

        fpath = os.path.join(wav_dirpath,str_)
        y, sr = sf.read(fpath)
        wav_data.append(y)
    wav_data = np.transpose(np.asanyarray(wav_data))

    return wav_data, sr


def extract_frames(y_stft, frame_size=10, feature_dim=257):
    all_stft = []
    for i in range(y_stft.shape[2]):
        if i>frame_size and i< y_stft.shape[2]-frame_size:
            t_future = y_stft[:,:,i:i+frame_size,:]
            t_past = y_stft[:,:,i-frame_size:i,:]
            #print(t_past.shape, t_future.shape, y_stft[:,:,i,:].unsqueeze(0).shape)
        elif i<frame_size and i < y_stft.shape[2]-frame_size:
            t_future = y_stft[:,:,i:i+frame_size,:]
            t_past = y_stft[:,:,:i,:]
            zeros = torch.zeros(1,y_stft.shape[1],frame_size-i,feature_dim)
            t_past = torch.cat([t_past, zeros.double()], dim=2)
            #print(10-i, i, t_past.shape, t_future.shape, y_stft[:,:,i,:].unsqueeze(0).shape)
        elif i == frame_size or i == y_stft.shape[2] - frame_size:
            t_future = y_stft[:,:,i:i+frame_size,:]
            t_past = y_stft[:,:,i-frame_size:i,:]

            #print(i, t_past.shape, t_future.shape, y_stft[:,:,i,:].unsqueeze(0).shape)
        else:
            t_future = y_stft[:,:,i:y_stft.shape[2],:]
            zeros = torch.zeros(1,y_stft.shape[1],i+frame_size-y_stft.shape[2],feature_dim)
            t_future = torch.cat([t_future, zeros.double()], dim=2)
            t_past = y_stft[:,:,i-frame_size:i,:]

        #print(t_past.shape, y_stft[:,:,i,:].unsqueeze(2).shape, t_future.shape)
        stft = torch.cat([t_past,y_stft[:,:,i,:].unsqueeze(2),t_future], dim=2)

        all_stft.append(stft.numpy())


    all_stft = np.asanyarray(all_stft)
    all_stft = torch.from_numpy(all_stft)

    train_stft = all_stft.squeeze(1)

    return train_stft


# In[2]:


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h








#import IPython.display as ipd

from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.preprocessing.data import _handle_zeros_in_scale


# In[45]:

import numpy as np
import librosa as lb
import scipy.linalg
import sys


eps = sys.float_info.epsilon
eta = 1e6
lambda_cor=0.95



def intern_filter(Rxx, Rnn, mu=1, type='r1-mwf', rank='Full'):
    """
    Computes a filter according to references [1] (SDW-MWF) or [2] (GEVD-MWF).

    :param Rxx: Speech covariance matrix
    :param Rnn: Noise covariance matrix
    :param mu: Speech distortion constant [default 1]
    :param type: (string) Type of filter to compute (SDW-MWF (see [1], equation (4)) or GEVD (see [2])). [default 'mwf']
    :param rank: ('Full' or 1) Rank-1 approximation of Rxx ? (see [2]). [default 'Full']
    :return:    - Wint: Filter coefficients
                - t1: (for GEVD case) vector selecting signals in GEVD fashion, to get correct reference signal
    """
    t1 = np.vstack((1, np.zeros((np.shape(Rxx)[0] - 1, 1))))    # Default is e1, selecting first column
    sort_index = None
    if type == 'r1-mwf':
        # # ----- Make Rxx rank 1  -----
        D, X = np.linalg.eig(Rxx)
        D = np.real(D)  # Must be real (imaginary part due to numerical noise)
        Dmax, maxind = D.max(), D.argmax()  # Find maximal eigenvalue
        Rxx = np.outer(np.abs(Dmax) * X[:, maxind],
                       np.conjugate(X[:, maxind]).T)  # Rxx is assumed to be rank 1
        # -----------------------------
        P = np.linalg.lstsq(Rnn, Rxx, rcond=None)[0]
        Wint = 1 / (mu + np.trace(P)) * P[:, 0]  # Rank1-SDWMWF (see [1])

    elif type == 'gevd':
        # TODO: inquire wether scipy.linalg.eig is much slower than scipy.linag.eigh
        D, Q = scipy.linalg.eig(Rxx, Rnn)               # GEV decomposition of Rnn^(-1)*Ryy
        D = np.maximum(D,
                       eps * np.ones(np.shape(D)))      # Prevent negative eigenvalues
        D = np.minimum(D,
                       eta * np.ones(np.shape(D)))      # Prevent infinite eigenvalues
        sort_index = np.argsort(D)                      # Sort the array to put GEV in descending order in the diagonal
        D = np.diag(D[sort_index[::-1]])                # Diagonal matrix of descending-order sorted GEV
        Q = Q[:, sort_index[::-1]]                      # Sorted matrix of generalized eigenvectors
        if rank != 'Full':                              # Rank-1 matrix of GEVD;
            D[rank:, :] = 0                             # Force zero values for all GEV but the highest
        # Filter
        Wint = np.matmul(Q,
                         np.matmul(D,
                                   np.matmul(np.linalg.inv(D + mu * np.eye(len(D))),
                                             np.linalg.inv(Q))))[:, 0]
        t1 = Q[:, 0] * np.linalg.inv(Q)[0, 0]
    elif type == 'basic':
        P = np.linalg.lstsq(mu*Rnn + Rxx, Rxx, rcond=None)[0]
        Wint = P[:, 0]

    else:
        raise AttributeError('Unknown filter reference')

    return Wint, (t1, sort_index)


def spatial_correlation_matrix(Rxx, x, lambda_cor=0.95, M=None):
    """
    Return spatial correlation matrix computed as exponentially smoothing of :
            - if M is None: x*x.T
                            so Rxx = lambda * Rxx + (1 - lambda)x*x.T
              x should then be an estimation of the signal of which one wants the Rxx

            - if M is not None: M*x*x.T
              x is then the mixture
    :param Rxx:             Previous estimation of Rxx
    :param x:               Signal (estimation of noise/speech if M is none; mixture otherwise)
    :param lambda_cor:      Smoothing parameter
    :param M:               Mask. If None, x is the estimation of the signal of which one wants the Rxx.
    :return: Rxx            Current eximation of Rxx
    """
    if M is None:
        Rxx = lambda_cor * Rxx + (1 - lambda_cor) * np.outer(x, np.conjugate(x).T)
    else:
        Rxx = lambda_cor * Rxx + M * (1 - lambda_cor) * np.outer(x, np.conjugate(x).T)
    return Rxx



def truncated_eye(N, j, k=0):
    """
    Create a NxN matrix with k consecutive ones in the diagonal.
    :param N:   (int) Dimension of output matrix
    :param j:   (int) Number of ones in the diagonal
    :param k:   (int) Diagonal in question (k>0 shifts the diagonal to a sub-diagonal)
    :return: A truncated eye matrix
    """
    v1 = np.ones((j, ))
    v0 = np.zeros((N - j, ))

    return np.diag(np.concatenate((v1, v0), axis=0), k=k)





# In[40]:
def masking(y, m, win_len=512, win_hop=256):
    y_stft = lb.core.stft(y, n_fft=win_len, hop_length=win_hop, center=True)

    m = np.pad(m, ((0, 0), (1, 1)), 'reflect')
    y_m = m*y_stft

    y_f = lb.core.istft(y_m, hop_length=win_hop, win_length=win_len, center=True, length=len(y))

    y_f = np.transpose(np.asanyarray([y_f,y_f,y_f]))

    return y_f

def multichannel_weiner_filter_current(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95, type_='basic'):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    w = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_out = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:
                w[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type=type_, rank=1)

            except np.linalg.linalg.LinAlgError:
                pass

            y_filt[i_freq, i_frame, :] = np.matmul(np.conjugate(w[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])


    for i_ch in range(n_ch):
        y_out[:, i_ch] = lb.core.istft(np.pad(y_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    
    return y_out


def multichannel_weiner_filter_custom(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    w_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    w_gevd = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_r1 = np.zeros(y.shape)
    y_filt_gevd = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_gevd = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:
                w_r1[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='r1-mwf', rank=1)
                w_gevd[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='gevd', rank=1)

            except np.linalg.linalg.LinAlgError:
                pass

            y_filt_r1[i_freq, i_frame, :] = np.matmul(np.conjugate(w_r1[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            y_filt_gevd[i_freq, i_frame, :] = np.matmul(np.conjugate(w_gevd[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])


    for i_ch in range(n_ch):
        y_r1[:, i_ch] = lb.core.istft(np.pad(y_filt_r1[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        y_gevd[:, i_ch] = lb.core.istft(np.pad(y_filt_gevd[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    
    return y_r1, y_gevd

def multichannel_weiner_filter_custom2(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')

    w_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_filt_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_r1 = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:

                w_r1[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='r1-mwf', rank=1)
                
            except np.linalg.linalg.LinAlgError:
                pass

            y_filt_r1[i_freq, i_frame, :] = np.matmul(np.conjugate(w_r1[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            

    for i_ch in range(n_ch):
        y_r1[:, i_ch] = lb.core.istft(np.pad(y_filt_r1[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    return y_r1




def len_match_to_nchannel(s_dry, s):
    if s_dry.shape[0] > s.shape[0]:
        s_dry = s_dry[:s.shape[0],]
    else:
        diff = s.shape[0] - s_dry.shape[0]
        #print(s_dry.shape, np.zeros(diff,).shape)
        s_dry = np.concatenate((s_dry, np.zeros(diff,)), axis=0)

    s_dry = np.concatenate(([s_dry], [s_dry], [s_dry]), axis=0)
    s_dry = np.transpose(s_dry)

    return s_dry

def mir_eval_noisy(x, n, x_dry, n_dry, y):

    x_dry = len_match_to_nchannel(x_dry, x)
    n_dry = len_match_to_nchannel(n_dry, n)

    reference_sources_noisy = np.vstack((x, n))
    dry_reference_sources_noisy = np.vstack((x_dry, n_dry))

    estimated_sources_noisy = np.vstack((y, y - y))

    sdr_n, isr, sir_n, sar_n, _ = mir_eval.separation.bss_eval_images(np.transpose(dry_reference_sources_noisy), 
                                                                  np.transpose(estimated_sources_noisy), compute_permutation=False)

    sdr_in, isr, sir_in, sar_in, _ = mir_eval.separation.bss_eval_images(np.transpose(reference_sources_noisy), 
                                                                  np.transpose(estimated_sources_noisy), compute_permutation=False)

    return (sdr_n[0], sir_n[0], sar_n[0]), (sdr_in[0], sir_in[0], sar_in[0])

    
    
def meanvar_normalise(data):
    dtype = data[0][0].dtype
    last_sample_count = 0
    mean_=0.0
    var_=0.0
    for idx, x in enumerate(data):
        #print(idx, x.shape)
        mean_, var_, _ = _incremental_mean_and_var(x, mean_, var_, last_sample_count)
        last_sample_count += len(x)
    mean_, var_ = mean_.astype(dtype), var_.astype(dtype)
    stddev_ = np.sqrt(var_)
    return mean_, var_, stddev_

def meanvar_scale(x, data_mean, data_stddev):
    return (x - data_mean) / _handle_zeros_in_scale(data_stddev, copy=False)

def meanvar_scaleback(x, data_mean, data_stddev):
    return x*data_stddev + data_mean


def scale_data(data, normparam, nch=3):
    for i in range(nch):
        data[i] = meanvar_scale(data[i], normparam['mean'], normparam['stddev'])
    return data

def scale_back_data(data, normparam, nch=3):
    for i in range(nch):
        data[i] = meanvar_scaleback(data[i], normparam['mean'], normparam['stddev'])
    return data

def ideal_binary_mask(x, n, thr=0):
    """Returns the ideal binary mask.
    Arguments:
    - x: speech spectrogram
    - n: noise spectrogram (same size as x)
    - thr: threshold value in dB [0]
    Output:
    -ibm: ideal binary mask - 0 or 1 in each bin
    """
    thr = db2lin(thr)
    xi = x / n
    ibm = (xi >= thr)
    return ibm


def wiener_mask(x, n, power=2):
    """Returns the ideal wiener mask.

    Arguments:
        - x:        speech spectrogram (real values)
        - n:        noise spectrogram (real values; same size as x)
        - power:    power of SNR in gain computation [2]
    Output:
        - wm: wiener mask values between 0 and 1
    """
    xi = (x / n) ** power
    wf = xi / (1 + xi)
    return wf

def ideal_mixture_mask(s_stft, n_stft, sum_of_squares=False):
    if sum_of_squares == True:
        Ms = np.abs(s_stft.astype('float'))/(np.abs(s_stft.astype('float')) + np.abs(n_stft.astype('float'))) # |S_w| / |S_w| + |N_w|
    else:
        Ms = np.abs(s_stft)/np.abs(s_stft + n_stft) # |S_w| / |(S_w + N_w)|

    return Ms





def mir_eval_all(s_dry, n_dry, y, y_out):

    min_len = min(y.shape[0], s_dry.shape[0], n_dry.shape[0])     
    
    dry_reference_sources = np.hstack((s_dry[:min_len, :], n_dry[:min_len, :]))
    
    estimated_sources_out1 = np.hstack((y_out[0][:min_len, :], y[:min_len, :] - y_out[0][:min_len, :]))

    eval_out1 = eval_mir(estimated_sources_out1, dry_reference_sources)

    estimated_sources_out2 = np.hstack((y_out[1][:min_len, :], y[:min_len, :] - y_out[1][:min_len, :]))

    eval_out2 = eval_mir(estimated_sources_out2, dry_reference_sources)

    
    return (eval_out1,eval_out2)

def mir_eval_single(s_dry, n_dry, y, y_out):

    min_len = min(y.shape[0], s_dry.shape[0], n_dry.shape[0])     
    
    dry_reference_sources = np.hstack((s_dry[:min_len, :], n_dry[:min_len, :]))
    
    estimated_sources_out1 = np.hstack((y_out[:min_len, :], y[:min_len, :] - y_out[:min_len, :]))

    eval_out1 = eval_mir(estimated_sources_out1, dry_reference_sources)

    
    return eval_out1



def multichannel_weiner_filter_mu(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    
    w_r1 = {}
    y_filt_r1 = {}
    y_r1 = {}
    
    for mu_ in [0.1, 0.001, 0.00001, 0]:
        w_r1[mu_] = np.zeros((n_freq, n_frames, n_ch), 'complex')
        y_filt_r1[mu_] = np.zeros((n_freq, n_frames, n_ch), 'complex')
        y_r1[mu_] = np.zeros(y.shape)
        
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:
                for mu_ in [0.1, 0.001, 0.00001, 0]:
                    w_r1[mu_][i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu_, type='r1-mwf', rank=1)
                
                
                
            except np.linalg.linalg.LinAlgError:
                pass
            
            for mu_ in [0.1, 0.001, 0.00001, 0]:
                y_filt_r1[mu_][i_freq, i_frame, :] = np.matmul(np.conjugate(w_r1[mu_][i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            
    for mu_ in [0.1, 0.001, 0.00001, 0]:
        for i_ch in range(n_ch):
            y_r1[mu_][:, i_ch] = lb.core.istft(np.pad(y_filt_r1[mu_][:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    return y_r1












def ideal_ratio_mask(s_stft, n_stft):
    #mask = np.square(np.abs(s_stft))/(np.square(np.abs(s_stft)) + np.square(np.abs(n_stft)))

    n_reg = np.maximum(abs(n_stft), sys.float_info.epsilon)
    ratio = (abs(s_stft) / n_reg)
    mask = ratio / (1 + ratio)
    
    return mask




def multichannel_weiner_filter_fasnet(y, f, n, mask_compute=None, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    assert y.shape == f.shape
    
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')

    w_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_filt_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_r1 = np.zeros(y.shape)
    y_mask = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)

    s_stft = stft_nchannel_wave(f)
    n_stft = stft_nchannel_wave(n)
    #y_stft = stft_nchannel_wave(y)
    
    if mask_compute == None:
        s_stft_hat = s_stft
        n_stft_hat = n_stft
        ms = None      
    else:
        ms = ideal_ratio_mask(s_stft, n_stft)
        ms = np.transpose(ms, (2,0,1))
        print(ms.shape, 'mask shape')# Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:

                w_r1[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='r1-mwf', rank=1)
                
            except np.linalg.linalg.LinAlgError:
                pass

            y_filt_r1[i_freq, i_frame, :] = np.matmul(np.conjugate(w_r1[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            

    for i_ch in range(n_ch):
        y_r1[:, i_ch] = lb.core.istft(np.pad(y_filt_r1[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))

    return y_r1


def multichannel_weiner_filter_gevd_basic(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')

    w_gevd = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt_gevd = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_gevd = np.zeros(y.shape)
    
    w_gevd_full = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt_gevd_full = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_gevd_full = np.zeros(y.shape)
    
    w_basic = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt_basic = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_basic = np.zeros(y.shape)
    
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:

                w_gevd[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='gevd', rank=1)
                w_gevd_full[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='gevd', rank='Full')
                w_basic[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='basic', rank=1)
                
            except np.linalg.linalg.LinAlgError:
                pass

            y_filt_gevd[i_freq, i_frame, :] = np.matmul(np.conjugate(w_gevd[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            y_filt_gevd_full[i_freq, i_frame, :] = np.matmul(np.conjugate(w_gevd_full[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            y_filt_basic[i_freq, i_frame, :] = np.matmul(np.conjugate(w_basic[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            

    for i_ch in range(n_ch):
        y_gevd[:, i_ch] = lb.core.istft(np.pad(y_filt_gevd[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        y_gevd_full[:, i_ch] = lb.core.istft(np.pad(y_filt_gevd_full[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        y_basic[:, i_ch] = lb.core.istft(np.pad(y_filt_basic[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        
    return y_gevd, y_gevd_full, y_basic



def multichannel_weiner_filter_gevd_full(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')

    w_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_filt_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_r1 = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:

                w_r1[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='gevd', rank='Full')
                
            except np.linalg.linalg.LinAlgError:
                pass

            y_filt_r1[i_freq, i_frame, :] = np.matmul(np.conjugate(w_r1[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            

    for i_ch in range(n_ch):
        y_r1[:, i_ch] = lb.core.istft(np.pad(y_filt_r1[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    return y_r1

def multichannel_weiner_filter_basic_full(y, ms, win_len=512, win_hop=256, mu=1, lambda_cor=0.95):
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')

    w_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_filt_r1 = np.zeros((n_freq, n_frames, n_ch), 'complex')
    
    y_r1 = np.zeros(y.shape)
    #print(y_stft.shape, ms.shape)
    
    # multichannel stft calculation, y_tf, s_tf, n_tf
    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(np.ascontiguousarray(y[:, i_ch]), n_fft=win_len, hop_length=win_hop, center=False)
        # Input estimation of signal and noise as s^ and n^
        s_stft_hat[:, :, i_ch] = ms[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - ms[i_ch]) * y_stft[:, :, i_ch]
    
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                               lambda_cor=lambda_cor_, M=None)

            try:

                w_r1[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                             mu=mu, type='basic', rank='Full')
                
            except np.linalg.linalg.LinAlgError:
                pass

            y_filt_r1[i_freq, i_frame, :] = np.matmul(np.conjugate(w_r1[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            

    for i_ch in range(n_ch):
        y_r1[:, i_ch] = lb.core.istft(np.pad(y_filt_r1[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                           hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    return y_r1



from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root


stft_options = dict(size=512, shift=128)
channels = 1
sampling_rate = 16000
delay = 3
iterations = 5
taps = 10
alpha=0.9999

def dereverb(y):
    Y = stft(y, **stft_options).transpose(2, 0, 1)
    Z = wpe(
        Y,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode='full'
    ).transpose(1, 2, 0)
    z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])

    return z.T[:,0]


