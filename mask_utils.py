import numpy as np
from code_utils.math_utils import db2lin
from code_utils.db_utils import stack_talkers
import librosa as lb
import sys

eps = sys.float_info.epsilon


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


def ideal_mixture_mask(x, n, power=2):
    """Return an ideal mask which, if applied to the mixture, returns the speech amplitude
    Arguments:
        - x:        Speech *complex* spectrogram
        - n:        Noise *complex* spectrogram
        - power:    power of amplitudes
    Ouput:
        - imm:      So-called ideal mixture mask
    """
    den = np.maximum(abs(x + n)**power, eps)

    imm = abs(x)**power / den
    return np.clip(imm, 0, 1)


def irm2ibm(irm, thr_db=0, type_irm=1):
    """
    Compute the IBM out of the given IRM.
    :param irm:         Ideal ratio mask IRM = SNR/(1+SNR)
    :param thr_db:      Threshold value in dB
    :param type_irm:    if 1: FFT amplitude are used to compute IRM; if 2: FFT power were used
    :return:            IBM = 1 where SNR > thr dB; 0 elsewhere
    """
    thr = db2lin(thr_db)
    thr **= type_irm  # Compensate for mask type
    return 1 * (irm > thr / (1 + thr) * np.ones(np.shape(irm)))


def error_between_masks(m1, m2):
    """
    Compute metrics to estimate the difference between input masks
    :param m1:
    :param m2:
    :return:
    """
    d = m1 - m2
    d2 = d ** 2
    # MSE
    err_mse = np.mean(np.mean(d2, axis=-1))
    # Mean positive error:
    err_mpe = np.mean(np.mean((d > 0) * d2))
    # Mean negative error:
    err_mne = np.mean(np.mean((d < 0) * d2))
    # Positive error ratio
    perr_rat = np.sum(np.sum(d > 0)) / m1.size
    # Negative error ratio
    nerr_rat = np.sum(np.sum(d < 0)) / m1.size

    return err_mse, err_mpe, err_mne, perr_rat, nerr_rat


# %%
def blur_mask(M, effect, blur_cst=0.1, avg_wgt=None, tlk_wavs=None, N=None):
    """

    :param M:               (array) Oracle mask
    :param effect:          (string) Temporal/Frequential smoothing or add SSN as a corruption of M
                            ['temp'/'freq'/'ssn']
    :param blur_cst         * smoothing constant if exponential smoothing
                            * number of frames if average smoothing
                            * if noise is added: the noise value will be in the range bin_value*(1 +- blur_cst)
    :param avg_wgt          Weights to apply on the frames/freq rows when averaging.
                            Should be 2*blur_cst+1 long if effect is 'time' or 'freq',
                             2*blur_cst+1 x 2*blur_cst+1 if effect is 'time-freq'
    :param tlk_wavs         List of wav files of talkers to compute SSN
    :return:
    """
    from pambox import distort

    if effect == "time_exp":
        M_blur = 1 * M
        for i_col in range(1, M.shape[1]):
            M_blur[:, i_col] = (1 - blur_cst) * M[:, i_col] + blur_cst * M_blur[:, i_col - 1]

    elif effect == "freq_exp":
        M_blur = 1 * M
        for i_row in range(1, M.shape[0]):
            M_blur[i_row, :] = (1 - blur_cst) * M[i_row, :] + blur_cst * M_blur[i_row - 1, :]

    elif effect == "time-freq_exp":
        M1 = 1 * M
        for i_col in range(1, M.shape[1]):
            M1[:, i_col] = (1 - blur_cst) * M[:, i_col] + blur_cst * M1[:, i_col - 1]
        M2 = 1 * M1
        for i_row in range(1, M.shape[0]):
            M2[i_row, :] = (1 - blur_cst) * M1[i_row, :] + blur_cst * M2[i_row - 1, :]
        M_blur = M2

    elif effect == 'time':
        M_blur = 1 * M
        n_cols = M.shape[1]
        for i_col in range(n_cols):
            M_blur[:, i_col] = np.average(M[:, np.maximum(0, i_col-blur_cst):np.minimum(i_col+blur_cst+1, n_cols)],
                                          axis=1, weights=avg_wgt)

    elif effect == 'freq':
        M_blur = 1 * M
        n_rows = M.shape[0]
        for i_row in range(n_rows):
            M_blur[i_row, :] = np.average(
                M[np.maximum(0, i_row - blur_cst):np.minimum(i_row + blur_cst + 1, n_rows), :],
                axis=0, weights=avg_wgt)

    elif effect == 'time-freq':
        M_blur = 1 * M
        n_cols = M.shape[1]
        n_rows = M.shape[0]
        for i_col in range(n_cols):
            for i_row in range(n_rows):
                M_blur[i_row, i_col] = np.average(
                    M[np.maximum(0, i_row - blur_cst[1]):np.minimum(i_row + blur_cst[1] + 1, n_rows),
                      np.maximum(0, i_col - blur_cst[0]):np.minimum(i_col + blur_cst[0] + 1, n_cols)], weights=avg_wgt)

    elif effect == "ssn":
        # Add speech-shaped noise
        dur_ssn = 11        # Duration of SSN time signal
        sig_ssn = "None"    # No need of this argument in the following use of stack_talkers
        n_fft = 512
        n_hop = 256
        # Create SSN
        tlk_tot, fs, _ = stack_talkers(tlk_wavs, dur_ssn, sig_ssn, 5)
        ssn = distort.noise_from_signal(tlk_tot, fs=fs, keep_env=False)  # SSN; length is longer than tar_dry
        # Get STFT
        ssn_stft = lb.core.stft(ssn, n_fft=n_fft, hop_length=n_hop, center=False)
        # Sum to mask
        ssn_mean = np.mean(np.mean(abs(ssn_stft)))
        m_mean = np.mean(np.mean(M))
        alpha = blur_cst * m_mean / ssn_mean     # Multiplication factor for interference
        M_blur = M + alpha * abs(ssn_stft[:, :M.shape[1]])
        M_blur = np.minimum(np.maximum(M_blur, np.zeros(M_blur.shape)), np.ones(M_blur.shape))  # Bound [0, 1]

    elif effect == 'wn':
        # Create uniformly distributed WN between -1 and 1
        wn = 2 * np.random.random(M.shape) - 1
        M_blur = M + blur_cst * wn

    elif effect == 'itf':
        # Add a proportion of the normalized spectrum of noise
        # Normalization of spectrum is performed along freq axis to better blur the VAD characteristics of the mask
        N = abs(N).T/np.max(abs(N), axis=1)
        N = N.T
        M_blur = M + blur_cst * N

    else:
        raise AttributeError("Effect should be 'time', 'freq', 'ssn', 'time-freq' or 'wn'")

    # Compute corresponding MSE
    err_mse, err_mpe, err_mne, p_err_rat, n_err_rat = error_between_masks(M_blur, M)

    return np.clip(M_blur, 0, 1), err_mse, err_mpe, err_mne, p_err_rat, n_err_rat

