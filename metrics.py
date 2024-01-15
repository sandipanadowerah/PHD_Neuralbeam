#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:20:35 2018

@author: nfurnon 
        Based on:
            - intern Matlab script SNR_weighted2
            - acoustics https://github.com/python-acoustics/python-acoustics
        
"""
import numpy as np
import scipy
from code_utils.math_utils import lin2db, db2lin


def snr(s, n, db=True):
    """ Computes the SNR     Arguments:
        - s_out     target signal
        - n_out     noise signal
        - db        (bool) Express delta_snr in dB ?
    Outputs:
        - snr       SNR
        
    If the signals include segments equal to zero (e.g. resulting from 
    zero padding), these segments will not be taken into account in level 
    computation
    """
    snr_ = np.var(s[s != 0])/np.var(n[n != 0])
    return db*lin2db(snr_) + (not db)*snr_


def delta_snr(s_out, n_out, s_in, n_in, db=True):
    """ Computes the SNR difference between output and input
    Arguments:
        - s_out     output target signal
        - n_out     output noise signal
        - s_in      input target signal
        - n_in      input noise signal
        - db        (bool) Express delta_snr in dB ?
    Outputs:
        - delta_snr = snr_out - snr_in in dB, snr_out/snr_in otherwise
        
    If the target signal includes segments equal to zero (e.g. resulting from 
    zero padding), these segments will not be taken into account in level 
    computation
    """
    snr_in = snr(s_in, n_in, True)
    snr_out = snr(s_out, n_out, True)
    d_snr = snr_out - snr_in
    return db*d_snr + (not db)*db2lin(d_snr, 1)


def sd(s_out, s_in, db=True):
    """ Computes the speech distortion
    Arguments:
        - s_out     output target signal
        - s_in      input target signal
        - db        (bool) Express delta_snr in dB ?
    Outputs:
        - sd = s_in - s_out in dB, s_in/s_out otherwise
        
    If the target signal includes segments equal to zero (e.g. resulting from 
    zero padding), these segments will not be taken into account in level 
    computation
    """
    sd = np.var(s_in[s_in != 0])/np.var(s_out[s_out != 0])
    return db*lin2db(sd) + (not db)*sd


def fw_snr(s, n, fs, vad_tar=None, vad_noi=None, clipping=1, db=True):
    """ Computes the SNR weighted by the speech intelligibility
    Arguments:
        - s             Target speech. First axis should be the time one.
        - n             Noise speech. First axis should be the time one.
        - fs            Sampling frequency
        - clipping      If '1' the not-yet-weighted SNRs are limited to the 
                        interval [-15, 25] dB. Default 1
        - db            (bool) Return output in dB ? [True]
    Output
        - fw_snr        Frequency weighted SNR
        - fw_snr_mean   Average fw_SNR
        - F             Center frequencies
    """
    from code_utils.sigproc_utils import third_octave_filterbank

    # Band importance function
    r = 2**(1/6)
    if fs/2 > 4500:
        I = np.array([83, 95, 150, 289, 440, 578, 653, 711, 818, 844, 882, 898, 868, 844, 771, 527, 364, 185])*1e-4
        F = np.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
                      8000])
        f2 = F*r
        N = np.sum(f2 < fs/2)
        F = F[:N]
        I = I[:N]
    else:
        I = np.array([128, 320, 320, 447, 447, 639, 639, 767, 959, 1182, 1214, 1086, 1086, 757])*1e-4
        F = np.array([200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000])
        f2 = F*r
        N = np.sum(f2 < fs/2)
        F = F[:N]
        I = I[:N]
        
    snr_var = np.zeros((N,))
    s_p = np.zeros((N,))
    n_p = np.zeros((N,))
    s_f, n_f = [], []
    # Calculation of the SNR values in the different bands
    b, a = third_octave_filterbank(F, fs, order=4)
    for i in np.arange(N):
        s_f.append(scipy.signal.lfilter(b[i, :], a[i, :], s, axis=0))
        n_f.append(scipy.signal.lfilter(b[i, :], a[i, :], n, axis=0))
        if vad_tar is None:
            s_p[i] = lin2db(np.var(s_f[i][s_f[i] != 0]))
        else:
            s_p[i] = lin2db(np.var(s_f[i][vad_tar != 0]))

        if vad_noi is None:
            n_p[i] = lin2db(np.var(n_f[i][n_f[i] != 0]))
        else:
            n_p[i] = lin2db(np.var(n_f[i][vad_noi != 0]))

        snr_var[i] = s_p[i] - n_p[i]
        
        if clipping:
            snr_var[i] = np.minimum(np.maximum(-15, snr_var[i]), 25)

    fqwt_snr = I/np.sum(I)*snr_var
    fw_snr_mean = np.sum(fqwt_snr)

    # Convert if necessary in dB
    fqwt_snr = db * fqwt_snr + (not db) * db2lin(fqwt_snr)
    fw_snr_mean = db * fw_snr_mean + (not db) * db2lin(fw_snr_mean)

    return fqwt_snr, fw_snr_mean, F


def seg_snr(s, n, win_len, win_hop, vad=None, axis=-1):
    """
    Compute the segmental SNR as SNR_seg = sum_M(10*log10(sum_Nm(var(s)/var(n))) on windows Nm where there is speech.
        Nm: m-th window in signal
        M: number of windows of size win_length in the signal, if shifted of win_hop samples
    :param s:               (vector) speech vector
    :param n:               (vector) noise vector
    :param win_len:         (int) window length
    :param vad:             (vector) VAD values for window.
    :param win_hop:         (int) shift between windows. Overlap is possible (win_hop < win_len)
    :param axis:            (int) axis over which split the signal. If s and n are 1-D, this has no influence
    :return: seg_snr        Segmental SNR in dB
    """
    from code_utils.sigproc_utils import sliding_window
    from code_utils.db_utils import frame_vad
    import sys
    eps = sys.float_info.epsilon

    snr_min = -15       # Floor SNR
    snr_max = 25        # Ceil SNR

    if len(s) != len(n):
        l_pad_s = np.maximum(len(n) - len(s), 0)
        l_pad_n = np.maximum(len(s) - len(n), 0)
        s = np.pad(s, (0, l_pad_s), mode='reflect')
        n = np.pad(n, (0, l_pad_n), mode='reflect')
        print('padded ' + str(l_pad_s) + ', ' + str(l_pad_n) + ' samples to s, n')

    sw = sliding_window(s, win_len, win_hop, axis=axis)
    nw = sliding_window(n, win_len, win_hop, axis=axis)

    sw_var = np.maximum(np.var(sw, axis=-1), eps)
    nw_var = np.maximum(np.var(nw, axis=-1), eps)

    if vad is None:     # Take all frames into account if no VAD
        batch_vad = np.ones(sw_var.shape)
    else:               # Downsample VAD to one sample per frame otherwise
        batch_vad = frame_vad(vad, win_len, win_hop)[:sw_var.shape[0]]

    snr_seg = batch_vad * np.clip(10 * np.log10(sw_var/nw_var), snr_min, snr_max)
    snr_seg = sum(snr_seg)/sum(batch_vad)

    return snr_seg


def reverb_ratios(x, rir, reverb_start=20, fs=16000):
    """
    Compute the direct-to-reverberant ratio as well as the signal-to-reverberation ratio as defined in [1] and [2]
    :param x:                   Signal
    :param rir:                 RIR
    :param reverb_start:        In ms, duration of the direct path (highest peak + 20ms by default).
    :param fs                   (int) sampling frequency.

    :return:
        - DRR           Direct to reverberant ratio (dB)
        - SRR           Signal to reverberation ratio  (dB)

    NB: This script assumes that the direct path corresponds to the highest peak, which is not always true.
    TODO: Detect all peaks and take first one as the one corresponding to the direct path

    [1] Naylor et al. Speech Dereverberation. Springer 2010 [page 38]
    [2] Eaton et al. Direct-to-reverberant ratio estimation using a null-steered beamformer. IEEE 2015.
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7177929
    """
    # DRR
    i_peak = np.argmax(rir)
    n_d = int(1e-3 * reverb_start * fs)             # Sample where reverberation starts
    h_d = rir[:i_peak + n_d]
    h_r = rir[i_peak + n_d:]

    drr = 10*np.log10(sum(h_d**2)/sum(h_r**2))

    # SRR
    x_d = np.convolve(x, h_d)
    x_r = np.convolve(x, h_r)
    srr = 10*np.log10(sum(x_d**2)/sum(x_r**2))

    return drr, srr


def fw_sd(s_out, s_in, fs, clipping=1, db=True):
    """ Computes the speech distortion weighted by the speech intelligibility
    Arguments:
    ----------
        - s_out         Target speech after processing. First axis should be the time one.
        - s_in          Target speech before processing. First axis should be the time one.
        - fs            Sampling frequency
        - clipping      If '1' the not-yet-weighted SNRs are limited to the 
                        interval [-15, 25] dB. Default 1
        - db            (bool) Return output in dB ? [True]
    Output
    ------
        - fw_sd         Frequency weighted SD in dB
        - fw_sd_mean    Average fw_SD
        - F             Center frequencies
        
    This probably does not make much sense since the signals (target) are 
    speech so are already somehow speech weighted
    """
    from code_utils.sigproc_utils import third_octave_filterbank

    # Change s_in name to avoid changing its value
    s_in_ = 1 * s_in
    # Compensate for different scaling of signals
    var_in = np.var(s_in_[s_in_ != 0])
    var_out = np.var(s_out[s_out != 0])
    scale = var_out / var_in
    # s_in_ *= np.sqrt(scale)          # s_in has same broadband variance as s_out

    # Band importance function
    r = 2**(1/6)
    if fs/2 > 4500:
        I = np.array([83, 95, 150, 289, 440, 578, 653, 711, 818, 844, 882, 898, 868, 844, 771, 527, 364, 185])*1e-4
        F = np.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
        f2 = F*r
        N = np.sum(f2 < fs/2)
        F = F[:N]
        I = I[:N]
    else:
        I = np.array([128, 320, 320, 447, 447, 639, 639, 767, 959, 1182, 1214, 1086, 1086, 757])*1e-4
        F = np.array([200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000])
        f2 = F*r
        N = np.sum(f2 < fs/2)
        F = F[:N]
        I = I[:N]
        
    sd_var = np.zeros((N,))
    out_p = np.zeros((N,))
    in_p = np.zeros((N,))
    
    # Calculation of the SD values in the different bands
    b, a = third_octave_filterbank(F, fs, order=4)
    for i in np.arange(N):
        out_f = scipy.signal.lfilter(b[i, :], a[i, :], s_out, axis=0)
        in_f = scipy.signal.lfilter(b[i, :], a[i, :], s_in_, axis=0)
        out_p[i] = lin2db(np.var(out_f[out_f != 0]))
        in_p[i] = lin2db(np.var(in_f[in_f != 0]))
        sd_var[i] = in_p[i] - out_p[i]
        
        if clipping:
            sd_var[i] = np.minimum(np.maximum(0, sd_var[i]), 25)

    fqwt_sd = I / np.sum(I) * sd_var
    fw_sd_mean = np.sum(fqwt_sd)
    # Convert if necessary in dB
    fqwt_sd = db * fqwt_sd + (not db) * db2lin(fqwt_sd)
    fw_sd_mean = db * fw_sd_mean + (not db) * db2lin(fw_sd_mean)

    return fqwt_sd, fw_sd_mean, F
