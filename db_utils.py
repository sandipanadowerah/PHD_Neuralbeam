#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import librosa as lb
from scipy.signal import fftconvolve
import soundfile as sf
import re
import os
import glob
from code_utils.metrics import fw_snr, snr
import pickle
#import ipdb


# %% Functions used to create DB

def frame_vad(vad, N=512, hop_size=256, rat=4):
    """
    Downsamples a VAD by returning 0s and 1s for frames of length hop_size samples.
    This aims to return frames corresponding to a STFT matrix, therefore the parameter N is also required.
    :param vad:         VAD to downsample
    :param N:           Length of STFT window (unused for VAD estimation otherwise)
    :param hop_size:    Length of frame over which one counts the occurrences of 1
    :param rat:         Ratio of N over which the whole frame is considered to contain speech.
                        If the number of occurrences if 1 in the frame exceeds N/rat, the whole frame is
                        considered to contain speech.
    :return: vad_frame  Downsampled VAD, with binary values per frame
    """
    vad_tmp = lb.util.frame(vad, N, hop_size)
    nb_va = np.sum(vad_tmp, axis=0)
    vad_o = 1*(nb_va > N / rat)
    return vad_o


def convolve_and_pad(x, rir, length):
    """Convolve x with rir and padds to `length` if resulting signal is shorter than `length`. Otherwise truncate it.
    Arguments:
        - x         First signal to convolve (vector)
        - rir       Second signal to convolve (vector)
        - length    In *samples*, length to pad to if len(y)<length
    output:
        - y         Convolved signal of length length
    """
    y = fftconvolve(x, rir, mode='full')  # fftconvolve much faster than np.convolve for arrays length > 500
    y = np.concatenate((y, np.zeros(np.max((length - len(y), 0), ))))
    return y[:length]


def read_random_part(wavs_list, dur_out):
    """ Load random dur_out seconds of a signal whose name is stored in
    wavs_list
    Arguments:
        - wavs_list     list of signals to randomly pick in
        - dur_out       expected duration of output signal
    Output:
        - y             signal randomly picked in wavs_list, during dur_out sec
        - fs            Sampling frequency of y
        - rnd_file      random file id picked
        - rnd_start     random sample of signal where y begins
    """
    rnd_file = np.random.randint(0, len(wavs_list))
    sig_lng, fs = sf.read(wavs_list[rnd_file])
    rnd_start = np.int((len(sig_lng) - dur_out * fs) * np.random.rand())
    y = sig_lng[rnd_start:rnd_start + np.int(dur_out * fs)]

    return y, fs, rnd_file, rnd_start


def stack_talkers(tlk_list, dur_min, speaker, nb_tlk=5):
    """Stacks talkers from tlk_list until dur_min is reached and number of
    speakers exceeds nb_tlk
    Arguments:
        - tlk_list       list of flac/wav files to pick talkers in $
        - dur_min       Minimal duration *in seconds* of the signal
    """
    i_tlk = 0
    tlk_tot = np.array([])
    str_files = str()
    fs = 16000

    while len(tlk_tot) < dur_min * fs or i_tlk < nb_tlk:  # At least 5 talkers' speech shape
        rnd_tmp = np.random.randint(0, len(tlk_list))  # random talker we are going to pick
        spk_tmp = re.split('/', tlk_list[rnd_tmp])[-1].split('-')[0]
        if spk_tmp != speaker:  # Don't take same speaker for SSN
            tlk_tmp, fs = sf.read(tlk_list[rnd_tmp])
            tlk_tot = np.hstack((tlk_tot, tlk_tmp))
            i_tlk += 1
            str_files = str_files + os.path.basename(tlk_list[rnd_tmp])[:-5] + '\n'

    return tlk_tot, fs, str_files


def increase_to_snr(x, n, snr_out, vad_tar=None, vad_noi=None, weight=False, fs=None):
    """Changing the *noise* level to ensure a SNR of snr_out dB.
    Arguments:
        - x         Target signal
        - n         Noise signal
        - snr_out   Desired output SNR (in dB)
        - vad       VAD of noise to compute level on non silent parts. Should be the same length as n.
        - weight    (bool) If True, weight SNR in freq domain
        -fs         (float) Sampling frequency. Only required if weight=True
    Output:
        - n_ where n_ level is such that var(x) - var(n) = 10**(snr/10)

    TO DO: transform this in function(s) depending on RIR and SNR and noise signal
    """
    if weight:
        _, snr_0, _ = fw_snr(x, n, fs, vad_tar=vad_tar, vad_noi=vad_noi)
        n_ = n * 10 ** ((snr_0 - snr_out) / 20)
    else:
        if vad_tar is not None:
            var_x = np.var(x[vad_tar != 0])
        else:
            var_x = np.var(x[x != 0])

        if vad_noi is not None:
            var_n = np.var(n[vad_noi != 0])
        else:
            var_n = np.var(n[n != 0])
        n_ = n * np.sqrt(10 ** (-snr_out / 10) * var_x / var_n)
    return n_


def compute_mel_spectrogram(x, n_fft=512, hop_length=256, power=2.0, n_mels=128):
    """Computes spectrogram with input characteristics
    """
    x_stft = lb.core.stft(x, n_fft=n_fft, hop_length=hop_length, center=False)
    x_spec = lb.feature.melspectrogram(S=x_stft, n_fft=n_fft,
                                       hop_length=hop_length,
                                       power=power, n_mels=n_mels)
    return x_spec


# %% Functions modifying the CSV files

def concat_files(set):
    """
    Concatenate CSV files of all the processes ran in parallel.
    :param set:     'test-reverb-May3' or 'train-reverb-May3'
    :return:
    """
    nb_rir = 100
    name_csv_root = '/home/nfurnon/Documents/data/mnt_g5k/dataset/LibriSpeech/audio/marjorette/' + set + '/LOG/'
    name_csv_var = [str(i) + '_log.csv' for i in np.arange(nb_rir)]
    name_csv_concat = name_csv_root + 'all_log.csv'

    with open(name_csv_concat, 'wt') as f_out:
        f_out_writer = csv.writer(f_out)
        for i_file, name in enumerate(name_csv_var):
            with open(name_csv_root + name, 'rt') as f_in:
                f_in_reader = csv.reader(f_in)
                for row in f_in_reader:
                    if i_file == 0 and row[0] != '':
                        f_out_writer.writerow(row)
                    if i_file > 0 and row[0] != 'id' and row[0] != '':
                        f_out_writer.writerow(row)
                    else:
                        continue


def save_measured_snr(dataset, noise='CHiME'):
    """
    Measure input SNR at all mics, according to noise 'noise'
    :param dataset: which dataset ? (train/test)
    :param noise:   which noise to consider to measure input SNR ? ['CHiME']
    :return: No return, a csv is created.
    """
    if dataset == 'test':
        dataset = 'test-reverb-May3'
    elif dataset == 'train':
        dataset = 'train-reverb-May3'

    dir_tar = '/home/iwv33/jab74/corpus/LibriSpeech/audio/marjorette/' + dataset + '/WAV/Target/'
    dir_noi = '/home/iwv33/jab74/corpus/LibriSpeech/audio/marjorette/' + dataset + '/WAV/Noise/'
    tars = glob.glob(dir_tar + '*.wav')
    tars_sorted = sorted(tars, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    nois = glob.glob(dir_noi + '*_' + noise + '_*.wav')
    nois_sorted = sorted(nois, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    snr_in = np.zeros((len(tars_sorted),))
    fw_snr_in = np.zeros((len(tars_sorted),))

    for i in np.arange(len(tars_sorted)):
        x, _ = sf.read(tars_sorted[i])
        n, fs = sf.read(nois_sorted[i])
        # snr_in[i] = snr(x, n)
        _, fw_snr_in[i], _ = fw_snr(x, n, fs)
        if i%int(len(tars_sorted)/10)==0:
            print(str(i) + '/' + str(len(tars_sorted)))

    # Save as CSV
    # snr_in_rsp = np.reshape(snr_in, (-1, 8))
    fw_snr_in_rsp = np.reshape(fw_snr_in, (-1, 8))
    # np.savetxt('test_input-snr.csv', snr_in_rsp, delimiter=';')
    np.savetxt('test_input-fw-snr.csv', fw_snr_in_rsp, delimiter=';')


# %% Functions reading the CSV files

def get_string_info(info_key, rir_id):
    """
    Returns the string content corresponding to the key 'info_key' in the config determined by RIR n° rir_id.
    :param info_key:    one of ['id', 'RIR_id', 'room_len', 'room_wid', 'room_hei',
                                'room_absorb', 'mics_xyz', 'ss_xyz', 'tar_file', 'tar_segment',
                                'SSN_files', 'CHiME_file', 'CHiME_start', 'interTalker_files',
                                'tv_file', 'tv_start', 'SNR', 'clip']
    :param rir_id:      (int) - number of RIR
    :return: value of the information stored in the CSV-file
    """

    if rir_id in np.arange(1, 10001):
        set_name = 'train-reverb-all'
    elif rir_id in np.arange(10001, 11001):
        set_name = 'test-reverb-all'
    else:
        print('RIR_id out of bounds')
        return

    csv_fieldnames = ['id', 'RIR_id', 'room_len', 'room_wid', 'room_hei',
                      'room_absorb', 'mics_xyz', 'ss_xyz', 'tar_file', 'tar_segment',
                      'SSN_files', 'CHiME_file', 'CHiME_start', 'interTalker_files',
                      'tv_file', 'tv_start', 'SNR_dry', 'clip',
                      'SNR_cnv_1', 'SNR_cnv_2', 'SNR_cnv_3', 'SNR_cnv_4',
                      'SNR_cnv_5', 'SNR_cnv_6', 'SNR_cnv_7', 'SNR_cnv_8']
    
    name_csv_file = '/home/nfurnon/Documents/dataset/LibriSpeech/audio/marjorette/' + set_name + '/LOG/all_log.csv'
    csv_reader = csv.reader(open(name_csv_file, 'r'))
    csv_list = list(csv_reader)
    if set_name == 'train-reverb-May3':
        string_content = csv_list[rir_id][csv_fieldnames.index(info_key)]
    else:
        string_content = csv_list[rir_id - 10000][csv_fieldnames.index(info_key)]

    return string_content


def strarr2num(string_content, out_dim=None):
    """
    Return positions of microphones and sources in numerical format.

    :param string_content:      Should be an array converted into a string. ex: '[[1 3 4] \n [4 5 6]]'
    :param out_dim              If not None, nb of columns of output array to reshape
    :return: out                values of string in an array;
    """
    out_ = []
    # Trim useless string
    s_ = string_content.lstrip('[[').rstrip(']]')
    lines = s_.split('\n')
    for line in lines:
        for cnt in line.split(' '):
            try:
                out_.append(float(cnt.split('[')[-1].split(']')[0]))
            except ValueError:
                continue
    out_array = np.array(out_)
    if out_dim is not None:
        # out_ is 1-D array of all coordinates -> reshape it
        out_array = out_array.reshape((-1, out_dim))

    return out_array


def get_info(info_key, rir_id):
    """
    Returns the content corresponding to the key 'info_key' in the config determined by RIR n° rir_id.
    If content is numerical, output is float type

    :param info_key:
    :param rir_id:
    :return:
    """
    num_vars = ['id', 'RIR_id', 'room_len', 'room_wid', 'room_hei',
                'room_absorb', 'mics_xyz', 'ss_xyz', 'tar_file', 'tar_segment',
                'SSN_files', 'CHiME_file', 'CHiME_start', 'interTalker_files',
                'tv_file', 'tv_start', 'SNR_dry', 'clip',
                'SNR_cnv_1', 'SNR_cnv_2', 'SNR_cnv_3', 'SNR_cnv_4',
                'SNR_cnv_5', 'SNR_cnv_6', 'SNR_cnv_7', 'SNR_cnv_8']

    s_cnt = get_string_info(info_key, rir_id)
    if info_key in ['mics_xyz', 'ss_xyz',
                    'SNR_cnv_1', 'SNR_cnv_2', 'SNR_cnv_3', 'SNR_cnv_4',
                    'SNR_cnv_5', 'SNR_cnv_6', 'SNR_cnv_7', 'SNR_cnv_8']:
        n_cnt = strarr2num(s_cnt)
    elif any(info_key in s for s in num_vars):
        n_cnt = float(s_cnt)
    else:
        n_cnt = s_cnt

    return n_cnt


def figure_conf(rir_id):
    """
    Plots the configuration if RIR rir_id in set set_name
    :param rir_id:
    :param set_name:
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    mics_pos = get_info('mics_xyz', rir_id)
    mics_pos = mics_pos.reshape((8, -1))
    ss_pos = get_info('ss_xyz', rir_id)
    ss_pos = ss_pos.reshape((2, -1))

    rt = get_info('room_absorb', rir_id)
    room_len = get_info('room_len', rir_id)
    room_wid = get_info('room_wid', rir_id)

    f = plt.figure()
    plt.plot(ss_pos[:, 0], ss_pos[:, 1], 'o')
    plt.plot(mics_pos[:, 0], mics_pos[:, 1], 'x')
    plt.xlim(-1, 7)
    plt.ylim(-1, 5)
    plt.gca().add_patch(Rectangle((0, 0), room_len, room_wid, fill=False, linewidth=3))

    # Legend nodes and sources
    plt.text(mics_pos[0, 0], mics_pos[0, 1] + 0.25, 'node 1')
    plt.text(mics_pos[4, 0], mics_pos[4, 1] + 0.25, 'node 2')
    plt.text(ss_pos[0, 0], ss_pos[0, 1] + 0.1, 'Target')
    plt.text(ss_pos[1, 0], ss_pos[1, 1] + 0.1, 'Noise')

    plt.text(0.5, room_wid + 0.5, 'RT60 = ' + str(round(rt, 2)))

    return f


def get_best_node(rir_id):
    """
    Returns best node according to measured fw_SNR of reverberated signals. The rir_id also determines the set.
    :param rir_id:      (int) RIR id. If rir_id in (1, 10000): train; if rir_id in (10001, 11000): test
    :return:            0 or 1 corresponding to node 1 or 2, depending on which one measured the highest fw_SNR at its
                        microphones
    """
    from code_utils.math_utils import db2lin

    fw_snr = np.zeros((8, ))
    node_fw_snr = np.zeros((2, ))
    for i in np.arange(len(fw_snr)):
        fw_snr[i] = db2lin(get_info('SNR_cnv_' + str(i+1), rir_id)[0])

    node_fw_snr[0] = np.mean(fw_snr[:4])
    node_fw_snr[1] = np.mean(fw_snr[5:])

    return np.argmax(node_fw_snr)


def get_best_nodes(set_name, on_cluster=False):
    """
    Same as get_best_node but for all RIR of dataset set_name. Avoids opening and closing 1000 times the same file
    :param set_name:    'test-rever-all' or 'test-reverb-mid' or ...
    :param on_cluster:  (bool) Is one working on cluster ? Influence on path of dataset 
    :return:            an array of 0 or 1 corresponding to best node argument
    """
    import pandas as pd

    if not on_cluster:
        file = '/home/nfurnon/Documents/data/mnt_g5k/dataset/LibriSpeech/audio/marjorette/' + set_name + '/LOG/all_log.csv'
    elif on_cluster == 'g5k':
        file = '/talc3/multispeech/calcul/users/nfurnon/dataset/LibriSpeech/audio/marjorette/' + set_name + '/LOG/all_log.csv'
    elif on_cluster == 'xplor':
        file = '/projects/iwv33/jab74/dataset/LibriSpeech/audio/marjorette/' + set_name + '/LOG/all_log.csv'

    data = pd.read_csv(file)
    maxs = []

    # Linear regression between snr_dry and snr_cnv or seg_snr showed that snr_cnv has higher correlation with snr_dry
    # than segSNR (r2 = 9.99, see Loria notebook on 02.09.2019)
    for val in data['SNR_cnv_chm'].values:
        val_num = strarr2num(val)
        val_mean = [np.mean(val_num[:4]), np.mean(val_num[4:])]
        arg_val_max = np.argmax(val_mean)
        maxs.append(arg_val_max)

    return maxs


def get_alpha(rir_id):
    """
    Returns the angle between the two sources
    :param rir_id:          (int) RIR id. If rir_id in (1, 10000): train; if rir_id in (10001, 11000): test
    :return:                alpha; in deg, the angle between the two sources
    """

    import cmath

    # Positions
    mics_pos = get_info('mics_xyz', rir_id)
    o_pos = [1/2*(mics_pos[0, 0] + mics_pos[4, 0]),            # x coordinate of the sensors centre
             1/2*(mics_pos[0, 1] + mics_pos[4, 1])]            # y coordinate of the sensors centre
    ss_pos = get_info('ss_xyz', rir_id)

    # Transform into polar system
    o_z = complex(o_pos[0], o_pos[1])
    o_e = abs(o_z) * cmath.exp(np.angle(o_z) * 1j)

    ss_z0 = complex(ss_pos[0, 0], ss_pos[0, 1])
    ss_e0 = abs(ss_z0) * cmath.exp(np.angle(ss_z0) * 1j)
    ss_z1 = complex(ss_pos[1, 0], ss_pos[1, 1])
    ss_e1 = abs(ss_z1) * cmath.exp(np.angle(ss_z1) * 1j)

    oss0 = o_e - ss_e0
    oss1 = o_e - ss_e1

    alpha = round((np.angle(oss0) - np.angle(oss1))*180/np.pi, 0)

    return np.minimum(abs(alpha), 360-abs(alpha))


def get_phis(rir_id):
    """
    Returns the angle between the two sources
    :param rir_id:          (int) RIR id. If rir_id in (1, 10000): train; if rir_id in (10001, 11000): test
    :return:  phi0, phi1    angles between each node centre and the sources
    """

    # Positions
    mics_pos = get_info('mics_xyz', rir_id)
    mics_pos = mics_pos.reshape((8, -1))
    ss_pos = get_info('ss_xyz', rir_id)
    ss_pos = ss_pos.reshape((2, -1))

    mic_centre_0 = np.array([np.mean(mics_pos[:4, 0]), np.mean(mics_pos[:4, 1])])
    mic_centre_1 = np.array([np.mean(mics_pos[4:, 0]), np.mean(mics_pos[4:, 1])])

    ss_0 = np.array([ss_pos[0, 0], ss_pos[0, 1]])
    ss_1 = np.array([ss_pos[1, 0], ss_pos[1, 1]])

    v00 = mic_centre_0 - ss_0
    v01 = mic_centre_0 - ss_1
    v10 = mic_centre_1 - ss_0
    v11 = mic_centre_1 - ss_1

    # Round and convert to degree
    phi0 = np.arccos(np.inner(v00, v01) / (np.linalg.norm(v00, ord=2) * np.linalg.norm(v01, ord=2))) * 180 / np.pi
    phi1 = np.arccos(np.inner(v10, v11) / (np.linalg.norm(v10, ord=2) * np.linalg.norm(v11, ord=2))) * 180 / np.pi
    # Wrap to [0, pi]
    phi0 = np.minimum(abs(phi0), 360 - abs(phi0))
    phi1 = np.minimum(abs(phi1), 360 - abs(phi1))

    return phi0, phi1


def get_all_phis(set_name):
    import pandas as pd

    file = '/home/nfurnon/Documents/data/mnt_g5k/dataset/LibriSpeech/audio/marjorette/' + set_name + '/LOG/all_log.csv'
    
    data = pd.read_csv(file)
    mics_poss = np.array(data['mics_xyz'])
    ss_poss = np.array(data['ss_xyz'])

    all_phis = np.zeros((len(mics_poss), 2))
    for i in range(len(mics_poss)): 
        mics_pos, ss_pos = mics_poss[i], ss_poss[i]
        mics_pos = strarr2num(mics_pos)
        mics_pos = mics_pos.reshape((8, -1))
        ss_pos = strarr2num(ss_pos)
        ss_pos = ss_pos.reshape((2, -1))
        mic_centre_0 = np.array([np.mean(mics_pos[:4, 0]), np.mean(mics_pos[:4, 1])])
        mic_centre_1 = np.array([np.mean(mics_pos[4:, 0]), np.mean(mics_pos[4:, 1])])

        ss_0 = np.array([ss_pos[0, 0], ss_pos[0, 1]])
        ss_1 = np.array([ss_pos[1, 0], ss_pos[1, 1]])

        v00 = mic_centre_0 - ss_0
        v01 = mic_centre_0 - ss_1
        v10 = mic_centre_1 - ss_0
        v11 = mic_centre_1 - ss_1

        # Round and convert to degree
        phi0 = np.arccos(np.inner(v00, v01) / (np.linalg.norm(v00, ord=2) * np.linalg.norm(v01, ord=2))) * 180 / np.pi
        phi1 = np.arccos(np.inner(v10, v11) / (np.linalg.norm(v10, ord=2) * np.linalg.norm(v11, ord=2))) * 180 / np.pi
        # Wrap to [0, pi]
        phi0 = np.minimum(abs(phi0), 360 - abs(phi0))
        phi1 = np.minimum(abs(phi1), 360 - abs(phi1))

        all_phis[i, 0] = phi0
        all_phis[i, 1] = phi1

    return all_phis


def get_oim_for_rirs(dir_res, rir_start, nb_rir, noise, d_type, f_type, best_nodes):
    """
    Return the OIM at best node for a configuration determined by noise and d_type and for all RIRs.
    """
    metrics, results = [], []
    doc = dir_res \
          + d_type + '/' + f_type + '/' \
          + 'results_danse_' + str(rir_start) + '-' + str(rir_start + nb_rir - 1) + '_' + noise + '.p'

    with open(doc, 'rb') as f:
        # Dictionary
        results_ = pickle.load(f)
        # Values in dictionary
        for key, value in results_.items():
            metrics.append(key)
            results.append(value)

    delta_snr_best = np.zeros((nb_rir,))
    delta_fw_snr_best = np.zeros((nb_rir,))
    fw_sd_best = np.zeros((nb_rir,))
    sdr_best = np.zeros((nb_rir,))
    sir_best = np.zeros((nb_rir,))
    sar_best = np.zeros((nb_rir,))
    stoi_best = np.zeros((nb_rir,))

    # Very long loop to grasp information in CSV file
    for i in np.arange(nb_rir):
        # best node values (is there a way to avoid the loop ?)
        delta_snr_best[i] = results[0][i, best_nodes[i]]
        delta_fw_snr_best[i] = results[1][i, best_nodes[i]]
        fw_sd_best[i] = results[2][i, best_nodes[i]]
        stoi_best[i] = results[3][i, best_nodes[i]]
        sdr_best[i] = results[4][i, best_nodes[i]]
        sir_best[i] = results[6][i, best_nodes[i]]
        sar_best[i] = results[5][i, best_nodes[i]]

    return delta_snr_best, delta_fw_snr_best, fw_sd_best, stoi_best, sdr_best, sir_best, sar_best


# %% Functions for the icassp20 test database (was not actually used for icassp2020)
def gather_icassp20_infos(on_cluster='local'):
    if on_cluster == 'local':
        path_to_data = '/home/nfurnon/Documents/data/mnt_g5k/dataset/LibriSpeech/audio/icassp20/test-reverb/'
    elif on_cluster == 'grid5000':
        path_to_data = '/home/nfurnon/dataset/LibriSpeech/audio/icassp20/test-reverb/'
    elif on_cluster == 'explor':
        path_to_data = '/projects/iwv33/jab74/dataset/LibriSpeech/audio/icassp20/test-reverb/'

    path_to_logs = path_to_data + 'LOG/'

    i_start = 10001
    i_stop = 11000

    # Initialize array gathering all the infos
    info = np.load(path_to_logs + str(i_start) + '_info.p')
    d = dict.fromkeys(info.keys())
    val_list = [[] for k in info.keys()]

    # Gather all infos into intialized array
    for i in np.arange(i_start, i_stop + 1):
        d_name = str(i) + '_info.p'
        a = np.load(path_to_logs + d_name)
        for i, k in enumerate(a.keys()):
            val_list[i].append(a[k])
    for i, k in enumerate(d.keys()):
        d[k] = val_list[i]

    return d


def get_icassp20_best_nodes(d=None, on_cluster='local'):
    nb_mics_per_node = 4
    if d is None:
        if on_cluster == 'local':
            path_to_data = '/home/nfurnon/Documents/data/mnt_g5k/dataset/LibriSpeech/audio/icassp20/test-reverb/'
        elif on_cluster == 'grid5000':
            path_to_data = '/home/nfurnon/dataset/LibriSpeech/audio/icassp20/test-reverb/'
        elif on_cluster == 'explor':
            path_to_data = '/projects/iwv33/jab74/dataset/LibriSpeech/audio/icassp20/test-reverb/'

        path_to_logs = path_to_data + 'LOG/'
        d = np.load(path_to_logs + 'all_infos.p')
    snrs = np.array(d['fw_snrs'])
    best_nodes = np.zeros((snrs.shape[0], ))
    for i in range(len(best_nodes)):
        # This should be generalized for any nb of nodes
        snr_i = [np.mean(snrs[i][:nb_mics_per_node]), np.mean(snrs[i][nb_mics_per_node:])]
        best_nodes[i] = np.argmax(snr_i)

    return best_nodes.astype(int)


def get_icassp20_phis(rir_id, d=None, on_cluster='local'):
    """
    Returns the angle between the two sources
    :param rir_id:          (int) RIR id. If rir_id in (1, 10000): train; if rir_id in (10001, 11000): test
    :param d:               dictionary of values. If none, will be loaded from single dictionaries
    :param on_cluster:      (string) Where are we working from ('local', 'grid5000', 'explor')
    :return:  phi0, phi1    angles between each node centre and the sources
    """

    if d is None:
        d = gather_icassp20_infos(on_cluster=on_cluster)

    rir_id -= 10001

    # Positions
    mics_pos = d['mics_xyz'][rir_id]
    sous_pos = d['sous_xyz'][rir_id]

    mic_centre_0 = np.array([np.mean(mics_pos[0, :4]), np.mean(mics_pos[1, :4])])
    mic_centre_1 = np.array([np.mean(mics_pos[0, 4:]), np.mean(mics_pos[1, 4:])])

    ss_0 = np.array([sous_pos[0, 0], sous_pos[0, 1]])
    ss_1 = np.array([sous_pos[1, 0], sous_pos[1, 1]])

    v00 = mic_centre_0 - ss_0
    v01 = mic_centre_0 - ss_1
    v10 = mic_centre_1 - ss_0
    v11 = mic_centre_1 - ss_1

    phi0 = np.arccos(np.inner(v00, v01)/(np.linalg.norm(v00, ord=2)*np.linalg.norm(v01, ord=2)))*180/np.pi
    phi1 = np.arccos(np.inner(v10, v11)/(np.linalg.norm(v10, ord=2)*np.linalg.norm(v11, ord=2)))*180/np.pi
    # Wrap to [0, pi]
    phi0 = np.minimum(abs(phi0), 360 - abs(phi0))
    phi1 = np.minimum(abs(phi1), 360 - abs(phi1))

    return phi0, phi1


def get_icassp20_allphis(d=None, on_cluster='local'):
    """
    Runs 1000 times get_icassp2°_phis to get phis of all RIRs.
    """
    if d is None:
        d = gather_icassp20_infos(on_cluster=on_cluster)

    phis0, phis1 = np.zeros((1000, )), np.zeros((1000, ))

    for rir_id in range(0, 1000):
        # Positions
        mics_pos = d['mics_xyz'][rir_id]
        sous_pos = d['sous_xyz'][rir_id]

        mic_centre_0 = np.array([np.mean(mics_pos[0, :4]), np.mean(mics_pos[1, :4])])
        mic_centre_1 = np.array([np.mean(mics_pos[0, 4:]), np.mean(mics_pos[1, 4:])])

        ss_0 = np.array([sous_pos[0, 0], sous_pos[0, 1]])
        ss_1 = np.array([sous_pos[1, 0], sous_pos[1, 1]])

        v00 = mic_centre_0 - ss_0
        v01 = mic_centre_0 - ss_1
        v10 = mic_centre_1 - ss_0
        v11 = mic_centre_1 - ss_1

        phi0 = np.arccos(np.inner(v00, v01)/(np.linalg.norm(v00, ord=2)*np.linalg.norm(v01, ord=2)))*180/np.pi
        phi1 = np.arccos(np.inner(v10, v11)/(np.linalg.norm(v10, ord=2)*np.linalg.norm(v11, ord=2)))*180/np.pi
        # Wrap to [0, pi]
        phi0 = np.minimum(abs(phi0), 360 - abs(phi0))
        phi1 = np.minimum(abs(phi1), 360 - abs(phi1))

        phis0[rir_id], phis1[rir_id] = phi0, phi1
    
    return phis0, phis1

