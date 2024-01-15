import numpy as np
import scipy
from code_utils.math_utils import next_pow_2


#%% VADs
def vad_ste(s, fs, thr=0.01, win_dur=0.05, hop_size=20):
    """ Voice activity detector for clear signal based on short time energy
    Arguments:
        - s         Signal
        - fs        Signal sampling frequency
        - thr       Threshold value. Ratio to maximal energy of the signal [0.01]
        - win_dur   Duration in sec of the window to compute energy [0.05]
        - hop_size  Hop size of windows (in samples (stupid)) [20]

    Output:
        - vad       VAD (1 if speech, 0 otherwise)
    """
    N = np.int(win_dur * fs)
    s2 = s ** 2
    E = np.zeros(len(s2))
    # unefficient loop to compute short term energy of the signal
    for n_ in np.arange((len(s2) - N) / hop_size + 1):
        n = np.int(n_)
        E[n * hop_size:np.minimum(n * hop_size + N, len(s2))] = sum(
            s2[n * hop_size:np.minimum(n * hop_size + N, len(s2))])

    E /= np.max(E)
    vad = 1 * (E > thr * np.max(E))
    return vad


def vad_oracle(x, thr=0.001):
    """ Oracle voice activity detector
    Arguments:
        - s         Signal
        - thr       Threshold value. Ratio to maximal energy of the signal [0.001]
        - N         Length of the window (in samples). [512]
        - hop_size  Hop size of windows (in samples) [256]

    Output:
        - vad       VAD (1 if speech, 0 otherwise)
    """
    x -= np.mean(x)
    x2 = x ** 2

    thr_ = thr * np.max(x2)

    vad_o = np.zeros(len(x2))
    vad_o[x2 > thr_] = 1

    return vad_o


def vad_oracle_batch(x_, N=512, hop_size=256, thr=0.001, rat=2):
    """ Oracle voice activity detector; speech is detected in batch mode, i.e. one binary decision is taken for a batch
    of N points.
    Arguments:
        - s         Signal
        - thr       Threshold value. Ratio to maximal energy of the signal [0.001]
        - N         Length of the window (in samples). [512]
        - hop_size  Hop size of windows (in samples) [256]
        - rat       Ratio of N. If N / rat samples are greater than thr * max(x**2),  the N samples are labelled as
                    speech. [2]

    Output:
        - vad       VAD (1 if speech, 0 otherwise). Length is len(x).
    """
    x = x_ - np.mean(x_)
    x2 = abs(x ** 2)

    thr_ = thr * np.quantile(x2, 0.99)      # Avoid determining threshold out of outliners
    vad_o = np.zeros(len(x2))
    # Buffer
    for n in np.arange(int(np.ceil((len(x2) - N) / hop_size + 1))):
        x2_win = x2[n * hop_size:np.minimum(n * hop_size + N, len(x2))]
        x2_win_va = 1 * (x2_win > thr_)
        nb_va = sum(x2_win_va)
        N_ = len(x2_win)        # Last window has probably less samples than all other ones
        if nb_va >= np.int(N_ / rat):
            vad_o[n * hop_size:np.minimum(n * hop_size + N, len(x2))] = 1
    return vad_o


#%% Filterbanks

def third_octave_filterbank(F, fs, order=8):
    """
    Returns filter num and denom of third-octave filterbank
    /!\ Suboptimal function, minimalist
    Arguments:
        - F         Center frequencies
        - fs        Sampling frequency
        - order     Order of the (butterworth) bandpass filter
    """
    from acoustics.signal import OctaveBand
    N = len(F)
    b = np.zeros((N, np.int(2 * order + 1)))
    a = np.zeros((N, np.int(2 * order + 1)))
    for i in np.arange(N):
        ob = OctaveBand(center=F[i], fraction=3)
        b[i, :], a[i, :] = scipy.signal.butter(order, np.array([ob.lower.item(), ob.upper.item()]) * 2 / fs,
                                               btype='bandpass', output='ba')

    return b, a


#%% Coming from other websites
def sliding_window(data, size, stepsize=1, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    NB: Slide over the last axis
    Parameters
    ----------
    data : np array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : np array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


def noise_from_signal(x):
    """Create a noise with same spectrum as the input signal.

    Parameters
    ----------
    x : array_like
        Input signal.

    Returns
    -------
    ndarray
        Noise signal.

    Copyright https://github.com/achabotl/pambox/blob/develop/pambox/distort.py
    """
    x = np.asarray(x)
    n_x = x.shape[-1]
    n_fft = next_pow_2(n_x)
    X = np.fft.rfft(x, next_pow_2(n_fft))
    # Randomize phase.
    noise_mag = np.abs(X) * np.exp(
        2 * np.pi * 1j * np.random.random(X.shape[-1]))
    noise = np.real(np.fft.irfft(noise_mag, n_fft))
    out = noise[:n_x]

    return out


#%% Smoothing
def smooth_exp(sig, alpha):
    """
    Exponential smoothing
        :param sig:     signal to smooth
        :param alpha:   smoothing constant
    """
    sig_smooth = 1*sig
    for i in range(1, len(sig)):
        sig_smooth[i] = alpha * sig[i] + (1 - alpha) * sig_smooth[i-1]
    return sig_smooth

