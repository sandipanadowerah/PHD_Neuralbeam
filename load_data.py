from keras.utils import Sequence
import sklearn.preprocessing
from code_utils.mask_utils import irm2ibm
from code_utils.misc_utils import get_node_from_channel
import numpy as np
import h5py
import ipdb

# Parameters
FS = 16000
DUR_MAX = 11
N_FFT = 512
HOP_SIZ = 256
THR = 0  # Threshold value in dB for binary mask computation out of IRM
WIN_LEN = 8
WIN_HOP = 4

ARR_GEO = np.array([4, 4])

STFT_MIN = 1e-6
STFT_MAX = 1e3


def normalize_seq(seq, method, mean_=0, std_=1, axis=None):
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
    if method == 'cent_normvar':
        # 0-centered, unitary variance using statistics of the whole dataset
        seq -= mean_[:, np.newaxis]     # Center
        seq /= std_[:, np.newaxis]      # Unit-variance
    elif method == 'skl':
        seq = sklearn.preprocessing.normalize(seq, norm='l1', axis=axis)
    elif method == 'norm_on_seq':
        seq -= np.mean(seq, axis=axis)[:, np.newaxis]
        seq /= np.std(seq, axis=axis)[:, np.newaxis]
    elif method == 'norm21':
        # Normalize to almost 1 but avoid dividing by outlier
        if axis == None:
            seq /= np.quantile(seq, 0.99)
        elif axis == 0:
            seq /= np.quantile(seq, 0.99, axis=axis)[np.newaxis, :]
        elif axis == 1:
            seq /= np.quantile(seq, 0.99, axis=axis)[:, np.newaxis]


    return seq


class SubWinSequence(Sequence):
    """
    Sequence returning a tuple of (input, output) for a NN. input is made of win_len long windows of STFT frames. output
    are the corresponding windows of masks.
    """

    def __init__(self, input_set, batch_size, idx_to_take, stage, win_len=WIN_LEN, win_hop=WIN_HOP,
                 norm_way='norm_on_seq', fr_to_take='last', **kwargs):
        """
        :param input_set        (tuple) Either:
                                    - 2-element list of names of HDF5 files gathering [0] input and [1] label. For train
                                    OR
                                    - Nx2 list pointing to the N files to load.
                                      input_set[:, 0] = input; input_set[:, 1] = label.
                                      This is for test_set
        :param batch_size       (int) batch size
        :param idx_to_take      (list) List of indexes of files to load from input_set (used to distinguish val from test)
        :param win_len          (int) Window length, in number of frames
        :param win_hop          (int) Hop size between windows (for train set only)
        :param norm_way         (str) Way to normalize the data
        """

        self.batch_size = batch_size
        self.idx_to_take = idx_to_take
        self.input_set = self.get_input_set(input_set)
        self.win_len = win_len
        self.win_hop = win_hop
        # Do we predict (=give back ordered windows) or not
        self.stage = stage.split('_')[0]
        self.test_set = ("test" in stage)
        self.dataset = stage.split("_")[-1]
        self.norm_way = norm_way
        self.fr_to_take = fr_to_take

        # Load the whole dataset once for all to enable quicker memory access
        self.seg_len = int(1 + np.floor((DUR_MAX * FS - N_FFT) / HOP_SIZ))
        self.in_data, self.gt_data = self.load_data_set()
        self.win_nb_seg = self._wins_per_seg()

        if not self.test_set:  # Keep track of subwindows to load, determined by the index of their first frame
            # First frame to consider in one segment
            if self.stage == 'train':  # train_stage: do not train on first silent second
                n_start = np.floor((1 * FS - N_FFT) / HOP_SIZ)
            else:
                n_start = 0
            # Last possible subwindow-first-frame in the segment
            n_stop = self.seg_len - self.win_len + 1
            # TODO: in test_stage, if win_hop != 1, the last frames are skipped.
            # TODO: n_stop should be self.seg_len - self.win_hop and the exceeding frames padded as in test_set

            # Vector of all possible first frames in one segment
            nb_seg = len(self.idx_to_take)          # Total nb of segments in dataset
            v_start = np.arange(n_start, n_stop)    # Vector start:stop
            # Vector of all possible first frames if all segments are concatenated
            a_start = np.outer(np.ones((1, nb_seg)), v_start)         # Tile v_start as many times as we have segments
            v_hop = self.seg_len * np.arange(nb_seg)            # Add seg_len at n_start at every next segment
            first_frames = (a_start + v_hop[:, np.newaxis]).astype(int)     # All first frames (cumulated indices)
            self.first_frames = first_frames.reshape((first_frames.size, ))
            if self.stage == 'train':
                np.random.shuffle(self.first_frames)                       # Take different first frames at next epoch

    def __len__(self):
        if not self.test_set:
            # We approximate the nb of windows by 1/win_hop of all possible windows (defined by their first frame)
            nb_win = len(self.first_frames) / self.win_hop
        else:
            # In the test set, all segments don't have the same length.
            # Th length of the Sequence is computed out of all loaded segments
            nb_win = np.cumsum(self.win_nb_seg)[-1]
        # Number of batches we get out of nb_win
        return int(np.ceil(nb_win / self.batch_size))

    def _wins_per_seg(self):
        """
        Returns the number of subwindows in the segments of the dataset.
        Only useful for test case, where segments don't have the same length
        """
        if self.test_set:
            win_nb_seg = []
            for in_win in self.in_data:
                win_loc = 1 + int(in_win.shape[0] - self.win_len)  # Nb of overlapping windows fitting in local segment
                win_nb_seg.append(win_loc)
        else:
            win_nb_seg = np.tile(self.seg_len - self.win_len + 1, len(self.idx_to_take))

        return win_nb_seg

    def get_input_set(self, input_set):
        input_set = [[input_set[0][0]], [input_set[1][0]]]      # [[input1], [output1]]
        return input_set

    def load_data_set(self):
        """
        Load the whole dataset
        """
        if self.norm_way == "cent_normvar":
            # Load mean and variance required by normalization
            noise = self.input_set[0][0][0].split('Mix-')[-1].split('_')[0]
            stats = np.load('../../../dataset/LibriSpeech/audio/' + self.dataset +
                            '/STFT/stats-lin_Mix-' + noise + '_Ch-1.npy')
            self.mean_ = stats[()]['mean']
            self.std_ = stats[()]['std']
        else:
            self.mean_ = None
            self.std_ = None
        
        if self.test_set:
            # Empty arrays
            in_data, gt_data = [], []

            # Fill them with data
            for idx in self.idx_to_take:
                in_seg = np.load(self.input_set[0][0][idx])['stft']
                in_seg = np.clip(abs(in_seg).astype('float'), STFT_MIN, STFT_MAX)  # Only interested in abs value
                in_seg = normalize_seq(in_seg, self.norm_way, mean_=self.mean_, std_=self.std_, axis=None)
                in_data.append(in_seg.T)
                if self.stage == 'predict':    # Load labels only for train/evaluate, useless to predict
                    gt_seg = np.empty(in_seg.shape)
                else:
                    gt_seg = np.load(self.input_set[1][0][idx])['mask']
                # Deal with NaNs
                gt_seg[np.isnan(gt_seg)] = 0
                gt_data.append(gt_seg.T)

            return in_data, gt_data

        else:
            n_seg = len(self.idx_to_take)
            n_freq = np.int(N_FFT/2 + 1)
            n_frames = int(1 + np.floor((DUR_MAX * FS - N_FFT) / HOP_SIZ))
            in_data, gt_data = np.zeros((n_seg, n_frames, n_freq), 'float32'), \
                               np.zeros((n_seg, n_frames, n_freq), 'float32')
            for i_idx, idx in enumerate(self.idx_to_take):
                in_seg = np.load(self.input_set[0][0][idx])['stft']
                in_seg = np.clip(abs(in_seg).astype('float'), STFT_MIN, STFT_MAX)   # Only interested in abs value
                in_seg = normalize_seq(in_seg, self.norm_way, mean_=self.mean_, std_=self.std_, axis=None)
                in_data[i_idx] = in_seg.T
                if self.stage != 'predict':
                    gt_seg = np.load(self.input_set[1][0][idx])['mask']
                    gt_seg[np.isnan(gt_seg)] = 0    # Deal with NaNs
                    gt_data[i_idx] = gt_seg.T

            return np.array(in_data), np.array(gt_data)

    def get_test_batch(self, item):
        """
        The subwindow to load is computed out of batch_item and the windows which were already loaded. We keep track of
        the latter with the cumulated sum of subwindows in a segment.
        :param item:
        :return:
        """
        win_nb_cum = np.cumsum([0] + self.win_nb_seg)   # Cumulated number of windows loaded
        n_tot = item * self.batch_size                  # hop_size is one in inference mode
        k = np.where(win_nb_cum > n_tot)[0][0] - 1      # Last segment where batch took values
        n = int(n_tot - win_nb_cum[k])                  # Index of the frame in the segment to load

        in_wins, gt_wins = self._fill_test_batch(k, n)

        return np.array(in_wins), np.array(gt_wins)

    def _fill_test_batch(self, k, n):
        """
        Separated from get_test_batch() to facilitate children use
        :param k:   Sequence index
        :param n:   Frame index in the sequence
        :return:
                - in_wins   Input subwindows
                - gt_wins   Outptu subwindows. Here only mask outputs. Could be tuple if several outputs (e.g. for MTL)
        """
        is_loaded = 0                                   # Flag to avoid loading multiple times one same segment
        in_wins, gt_wins = [], []                       # Variables storing the sub-windows of the batch

        # Append sub-windows to vars. The last batch is very probably smaller than the other ones
        while len(in_wins) < self.batch_size and k < len(self.idx_to_take):
            # Load the k-th segment
            if not is_loaded:
                in_seg = self.in_data[k]
                gt_seg = self.gt_data[k]
                is_loaded = 1
                seg_len = in_seg.shape[0]

            # Store the sub-windows
            if n <= seg_len - self.win_len:  # Case where next sub-windows fits fully in rest of sequence
                in_win = in_seg[n:n + self.win_len, :]
                if self.fr_to_take == 'all':
                    gt_win = gt_seg[n:n + self.win_len, :]
                else:
                    if self.fr_to_take == 'mid':
                        frame_to_take = int(n + np.floor(self.win_len / 2))
                    elif self.fr_to_take == 'last':
                        frame_to_take = n + self.win_len - 1
                    else:
                        raise ValueError('Unknown fr_to_take value')
                    # gt_win = gt_seg[:, frame_to_take: frame_to_take + 1]    # Force dimension
                    gt_win = gt_seg[frame_to_take, :]

                in_wins.append(in_win)
                gt_wins.append(gt_win)
                n += self.win_hop

            elif seg_len - self.win_len < n < seg_len - self.win_len + self.win_hop:
                raise ValueError('The programmer did not investigate this eventuality; '
                                 'please set win_hop to 1 for test set')
            else:
                k += 1
                n = 0
                is_loaded = 0

        return in_wins, gt_wins

    def get_train_batch(self, item):
        in_wins, gt_wins = [], []
        # How many windows have we loaded depending on item
        win_idx = item * self.batch_size

        while len(in_wins) < self.batch_size and win_idx < len(self.first_frames) / self.win_hop:
            # Segment corresponding to loaded_wins
            k = int(np.floor(self.first_frames[win_idx] / self.seg_len))
            # Corresponding starting frame
            loc_frame = self.first_frames[win_idx] % self.seg_len
            # Load subwindows
            in_loc_win, gt_loc_win = self._get_train_subwindow(k, loc_frame)
            in_wins.append(in_loc_win)
            gt_wins.append(gt_loc_win)

            # Update params
            win_idx += 1

        return np.array(in_wins), np.array(gt_wins)

    def _get_train_subwindow(self, k, n):
        """
        We seperate this into a subfunction for easier use inherited sequences
        :param k:
        :param n:
        :return:
        """
        in_win = self.in_data[k, n:n + self.win_len, :]
        if self.fr_to_take == 'all':
            gt_win = self.gt_data[k, n:n + self.win_len, :]
        else:
            if self.fr_to_take == 'mid':
                frame_to_take = int(n + np.floor(self.win_len / 2))
            elif self.fr_to_take == 'last':
                frame_to_take = n + self.win_len - 1
            else:
                raise ValueError('Unknown fr_to_take value')
            # gt_win = self.gt_data[k, frame_to_take: frame_to_take + 1, :]
            gt_win = self.gt_data[k, frame_to_take, :]

        return in_win, gt_win

    def __getitem__(self, item):
        if not self.test_set:
            in_batch, gt_batch = self.get_train_batch(item)
        else:
            in_batch, gt_batch = self.get_test_batch(item)
        return np.array(in_batch), np.array(gt_batch)

    def on_epoch_end(self):
        if self.stage == 'train':
            np.random.shuffle(self.first_frames)


class CNNSequence(SubWinSequence):
    """
    Child class of SubWinSequence where input is STFT and Zcomp stacked over the channels. Output is unchanged.
    """
    def __init__(self, *args, **kwargs):
        self.z_axis = kwargs.get('z_axis', 3)       # stack z_comp as second channel [z_axis=3] or on freq axis [2]?
        SubWinSequence.__init__(self, *args, **kwargs)

    def get_input_set(self, input_set):
        return [[input_set[0][0], input_set[0][1]], [input_set[1][0]]]

    def load_data_set(self):
        if self.test_set:
            # Empty arrays
            in_data, gt_data = [], []
        else:
            n_seg = len(self.idx_to_take)
            n_freq = np.int(N_FFT/2 + 1)
            n_frames = int(1 + np.floor((DUR_MAX * FS - N_FFT) / HOP_SIZ))
            if self.z_axis == 1:
                in_data, gt_data = np.zeros((n_seg, n_frames, 2 * n_freq), 'float32'),\
                                   np.zeros((n_seg, n_frames, n_freq), 'float32')
            elif self.z_axis == 2:
                in_data, gt_data = np.zeros((n_seg, 2 * n_frames, n_freq), 'float32'),\
                                   np.zeros((n_seg, n_frames, n_freq), 'float32')
            elif self.z_axis == 3:
                in_data, gt_data = np.zeros((n_seg, n_frames, n_freq, 2), 'float32'),\
                                   np.zeros((n_seg, n_frames, n_freq), 'float32')
            else:
                raise ValueError('z_axis should be 1 (stack over frequency dimension), 2 (over time dimension) '
                                 'or 3 (over channels)')

        # Fill them with data
        for i_idx, idx in enumerate(self.idx_to_take):
            # Load STFT data
            in_s = np.load(self.input_set[0][0][idx])['stft']
            in_s = np.clip(abs(in_s).astype('float'), STFT_MIN, STFT_MAX)  # Only interested in abs value
            in_s = normalize_seq(in_s, self.norm_way, axis=None)

            # Load Zcomp data
            in_z = np.load(self.input_set[0][1][idx])
            in_z = np.clip(abs(in_z).astype('float'), STFT_MIN, STFT_MAX)  # Only interested in abs value
            in_z = normalize_seq(in_z, self.norm_way, axis=None)
            if in_z.shape != in_s.shape:    # DANSE returns Z as long as .wav signal and masks were padded
                in_z = np.pad(in_z, ((0, 0), (0, in_s.shape[1] - in_z.shape[1])), 'constant', constant_values=0)

            if self.z_axis >= len(in_s.shape) + 1:
                stacked_input = np.stack((in_s, in_z), axis=int(self.z_axis - 1))
            else:
                stacked_input = np.concatenate((in_s, in_z), axis=int(self.z_axis - 1))
            stacked_input = np.swapaxes(stacked_input, 0, 1)        # time x freq ( x  ch)

            if self.test_set:
                in_data.append(stacked_input)
            else:
                in_data[i_idx] = stacked_input

            # Load labels
            if self.stage != 'predict':     # Labels are useless in prediction
                gt_seg = np.load(self.input_set[1][0][idx])['mask']
            else:
                gt_seg = np.empty(in_s.shape)
            # Deal with NaNs
            gt_seg[np.isnan(gt_seg)] = 0
            if self.test_set:
                gt_data.append(gt_seg.T)
            else:
                gt_data[i_idx, :, :] = gt_seg.T

        return in_data, gt_data

    def _fill_test_batch(self, k, n):
        """
        Fill batch with stacked inputs (STFT and Zcomp) and with output (mask)
        :param k:
        :param n:
        :return:
        """
        is_loaded = 0                                   # Flag to avoid loading multiple times one same segment
        in_wins, gt_wins = [], []                       # Variables storing the sub-windows of the batch

        # Append sub-windows to vars. The last batch is very probably smaller than the other ones
        while len(in_wins) < self.batch_size and k < len(self.idx_to_take):
            # Load the k-th segment
            if not is_loaded:
                in_seg = self.in_data[k]
                gt_seg = self.gt_data[k]
                is_loaded = 1
                seg_len = in_seg.shape[0]

            # Store the sub-windows
            if n <= seg_len - self.win_len:  # Case where next sub-windows fits fully in rest of sequence
                in_win = in_seg[n:n + self.win_len]

                if self.fr_to_take == 'all':
                    gt_win = gt_seg[n:n + self.win_len, :]
                else:
                    if self.fr_to_take == 'mid':
                        frame_to_take = int(n + np.floor(self.win_len / 2))
                    elif self.fr_to_take == 'last':
                        frame_to_take = n + self.win_len - 1
                    else:
                        raise ValueError('Unknown fr_to_take value')
                    # gt_win = gt_seg[:, frame_to_take: frame_to_take + 1]    # Force dimension
                    gt_win = gt_seg[frame_to_take, :]

                in_wins.append(in_win)
                gt_wins.append(gt_win)
                n += self.win_hop

            elif seg_len - self.win_len < n < seg_len - self.win_len + self.win_hop:
                raise ValueError('The programmer did not investigate this eventuality; '
                                 'please set win_hop to 1 for test set')
            else:
                k += 1
                n = 0
                is_loaded = 0

        return in_wins, gt_wins         # in_wins is batch_size x n_frames x n_freq x n_ch

    def _get_train_subwindow(self, k, n):
        in_win = self.in_data[k, n:n + self.win_len]

        if self.fr_to_take == 'all':
            gt_win = self.gt_data[k, n:n + self.win_len, :]
        else:
            if self.fr_to_take == 'mid':
                frame_to_take = int(n + np.floor(self.win_len / 2))
            elif self.fr_to_take == 'last':
                frame_to_take = n + self.win_len - 1
            else:
                raise ValueError('Unknown fr_to_take value')
            gt_win = self.gt_data[k, frame_to_take, :]

        return in_win, gt_win


class AudioUnetSequence(CNNSequence):
    def __init__(self, *args, **kwargs):
        if kwargs['fr_to_take'] == 'all':
            raise ValueError("AudioUnet returns a single frame, argument `fr_to_take` shoud be 'mid' or 'last'")
        else:
            CNNSequence.__init__(self, *args, **kwargs)

    def __getitem__(self, item):
        in_data, gt_data = CNNSequence.__getitem__(self, item)
        # AudioUnet requires other convention 
        gt_unet = gt_data[:, :256, np.newaxis, np.newaxis]
        return np.swapaxes(in_data[:, :, :256, :], 1, 2), gt_unet

class OneChCNNSequence(SubWinSequence):
    def __init__(self, *args, **kwargs):
        SubWinSequence.__init__(self, *args, **kwargs)

    def __getitem__(self, item):
        in_data, gt_data = SubWinSequence.__getitem__(self, item)
        return in_data[:, :, :, np.newaxis], gt_data     # Return only STFT but also in 4-D format


class OneChAudioUnetSequence(OneChCNNSequence):
    def __init__(self, *args, **kwargs):
        if kwargs['fr_to_take'] == 'all':
            raise ValueError("AudioUnet returns a single frame, argument `fr_to_take` shoud be 'mid' or 'last'")
        else:
            OneChCNNSequence.__init__(self, *args, **kwargs)

    def __getitem__(self, item):
        in_data, gt_data = OneChCNNSequence.__getitem__(self, item)
        gt_unet = gt_data[:, :256, np.newaxis, np.newaxis]
        return np.swapaxes(in_data[:, :, :256, :], 1, 2), gt_unet


class ConstantWinsSequence(SubWinSequence):
    def __init__(self, *args, **kwargs):
        SubWinSequence.__init__(self, *args, **kwargs)

    def load_data_set(self):
        """
        Load the whole dataset but by padding the shorter test data with zero to reach the constant length of the
        training data (as required e.g. by Heymann which takes the whole sequence as input)
        """
        if self.norm_way == "cent_normvar":
            # Load mean and variance required by normalization
            noise = self.input_set[0][0][0].split('Mix-')[-1].split('_')[0]
            stats = np.load('../../../dataset/LibriSpeech/audio/' + self.dataset +
                            '/STFT/stats-lin_Mix-' + noise + '_Ch-1.npy')
            self.mean_ = stats[()]['mean']
            self.std_ = stats[()]['std']
        else:
            self.mean_ = None
            self.std_ = None

        if self.test_set:
            # Empty arrays
            in_data, gt_data = [], []

            # Fill them with data
            for idx in self.idx_to_take:
                in_seg = np.load(self.input_set[0][0][idx])['stft']
                in_seg = np.clip(abs(in_seg).astype('float'), STFT_MIN, STFT_MAX)  # Only interested in abs value
                in_seg = normalize_seq(in_seg, self.norm_way, mean_=self.mean_, std_=self.std_, axis=None)
                in_data.append(in_seg.T)
                if self.stage == 'predict':  # Load labels only for train/evaluate, useless to predict
                    gt_seg = np.empty(in_seg.shape)
                else:
                    gt_seg = np.load(self.input_set[1][0][idx])['mask']
                # Deal with NaNs
                gt_seg[np.isnan(gt_seg)] = 0
                gt_data.append(gt_seg.T)

            return in_data, gt_data

        else:
            n_seg = len(self.idx_to_take)
            n_freq = np.int(N_FFT / 2 + 1)
            n_frames = int(1 + np.floor((DUR_MAX * FS - N_FFT) / HOP_SIZ))
            in_data, gt_data = np.zeros((n_seg, n_frames, n_freq), 'float32'), \
                               np.zeros((n_seg, n_frames, n_freq), 'float32')
            for i_idx, idx in enumerate(self.idx_to_take):
                in_seg = np.load(self.input_set[0][0][idx])['stft']
                in_seg = np.clip(abs(in_seg).astype('float'), STFT_MIN, STFT_MAX)  # Only interested in abs value
                in_seg = normalize_seq(in_seg, self.norm_way, mean_=self.mean_, std_=self.std_, axis=None)
                in_data[i_idx] = in_seg.T
                if self.stage != 'predict':
                    gt_seg = np.load(self.input_set[1][0][idx])['mask']
                    gt_seg[np.isnan(gt_seg)] = 0  # Deal with NaNs
                    gt_data[i_idx] = gt_seg.T

            return np.array(in_data), np.array(gt_data)


class MTLSubWinSequence(SubWinSequence):
    def __init__(self, *args, **kwargs):
        SubWinSequence.__init__(self, *args, **kwargs)
        self.gt_mask, self.gt_vad = self.gt_mask

    def get_input_set(self, input_set):
        return [[input_set[0][0]], [input_set[1][0], input_set[1][1]]]     # [[input1], [output1, output2]]

    def load_data_set(self):
        """
        Load the whole dataset
        """
        if self.norm_way == "cent_normvar":
            # Load mean and variance required by normalization
            inter_noise = self.input_set[0][0].split('Mix-')[-1].split('_')[0]
            stats = np.load(
                '../../../dataset/LibriSpeech/audio/' + self.dataset + '/STFT/stats-lin_Mix-'
                + inter_noise + '_Ch-1.npy')
            self.mean_ = stats[()]['mean']
            self.std_ = stats[()]['std']
        else:
            self.mean_ = None
            self.std_ = None

        # Empty arrays
        in_data, gt_mask, gt_vad = [], [], []

        # Fill them with data
        for idx in self.idx_to_take:
            in_seg = np.load(self.input_set[0][0][idx])['stft']
            in_seg = np.clip(abs(in_seg).astype('float'), STFT_MIN, STFT_MAX)  # Only interested in abs value
            in_seg = normalize_seq(in_seg, self.norm_way, mean_=self.mean_, std_=self.std_, axis=None)
            in_data.append(in_seg)

            if self.stage != 'predict':  # In train/evaluate case, load label. Useless otherwise
                mask_seg = np.load(self.input_set[1][0][idx])['mask']
                vad_seg = np.load(self.input_set[1][0][idx])   # Load VAD file
                vad_seg = np.concatenate((vad_seg, np.zeros((mask_seg.shape[-1] - len(vad_seg)))),
                                         axis=0)    # Concatenate because VAD were not padded before saving
            else:
                mask_seg = np.empty(in_seg.shape)
                vad_seg = np.empty(in_seg.shape[1])
            # Append output
            mask_seg[np.isnan(mask_seg)] = 0    # Deal with NaNs
            gt_mask.append(mask_seg)
            gt_vad.append(vad_seg[np.newaxis, :])

        if self.test_set:
            return in_data, gt_mask, gt_vad
        else:
            return np.array(in_data), (np.array(gt_mask), np.array(gt_vad))

    def get_train_batch(self, item):
        in_wins, mask_wins, vad_wins = [], [], []
        # How many windows have we loaded depending on item
        win_idx = item * self.batch_size

        while len(in_wins) < self.batch_size and win_idx < len(self.first_frames) / self.win_hop:
            # Segment corresponding to loaded_wins
            k = int(np.floor(self.first_frames[win_idx] / self.seg_len))
            # Corresponding starting frame
            loc_frame = self.first_frames[win_idx] % self.seg_len
            # Load subwindows
            in_loc_win, mask_loc_win, vad_loc_win = self._get_train_subwindow(k, loc_frame)
            in_wins.append(in_loc_win)
            mask_wins.append(mask_loc_win)
            vad_wins.append(vad_loc_win)

            # Update params
            win_idx += 1

        return np.array(in_wins), np.array(mask_wins), np.array(vad_wins)

    def _get_train_subwindow(self, k, n):
        if self.gt_frm == 'all':
            in_win = self.in_data[k, :, n:n + self.win_len]
            mask_win = self.gt_mask[k, :, n:n + self.win_len]
            vad_win = self.gt_vad[k, 0:1, n:n + self.win_len]
        else:
            if self.gt_frm == 'mid':
                frame_to_take = int(n + np.floor(self.win_len / 2))
            elif self.gt_frm == 'last':
                frame_to_take = n + self.win_len - 1
            else:
                raise ValueError('Unknown frame reference')
            in_win = self.in_data[k, :, frame_to_take: frame_to_take + 1]
            mask_win = self.gt_mask[k, :, frame_to_take: frame_to_take + 1]
            vad_win = self.gt_vad[k, 0:1, frame_to_take: frame_to_take + 1]

        return in_win.T, mask_win.T, vad_win.T

    def get_test_batch(self, item):
        """
        The subwindow to load is computed out of batch_item and the windows which were already loaded. We keep track of
        the latter with the cumulated sum of subwindows in a segment.
        :param item:
        :return:
        """
        win_nb_cum = np.cumsum([0] + self.win_nb_seg)   # Cumulated number of windows loaded
        n_tot = item * self.batch_size                  # hop_size is one in inference mode
        k = np.where(win_nb_cum > n_tot)[0][0] - 1      # Last segment where batch took values
        n = int(n_tot - win_nb_cum[k])                  # Index of the frame in the segment to load

        is_loaded = 0                                   # Flag to avoid loading multiple times one same segment
        in_wins, mask_wins, vad_wins = [], [], []       # Variables storing the sub-windows of the batch

        # Append sub-windows to vars. The last batch is very probably smaller than the other ones
        while len(in_wins) < self.batch_size and k < len(self.idx_to_take):
            # Load the k-th segment
            if not is_loaded:
                in_seg = self.in_data[k]
                mask_seg = self.gt_mask[k]
                vad_seg = self.gt_vad[k]
                is_loaded = 1
                seg_len = in_seg.shape[1]

            # Store the sub-windows
            if n <= seg_len - self.win_len:  # Case where next sub-windows fits fully in rest of sequence
                in_win = in_seg[:, n:n + self.win_len]
                mask_win = mask_seg[:, n:n + self.win_len]
                vad_win = vad_seg[n:n + self.win_len]
                in_wins.append(in_win.T)
                mask_wins.append(mask_win.T)
                vad_wins.append(vad_win.T)
                n += self.win_hop
            elif seg_len - self.win_len < n < seg_len - self.win_len + self.win_hop:
                # Case where full window does not fit in rest of sequence.
                in_win = in_seg[:, n:seg_len]
                mask_win = mask_seg[:, n:seg_len]
                vad_win = vad_seg[n:seg_len]
                # Pad to get shape expected by NN
                in_win = np.concatenate((in_win, np.zeros((in_win.shape[0], self.win_len - in_win.shape[1]))), axis=1)
                mask_win = np.concatenate((mask_win, np.zeros((mask_win.shape[0], self.win_len - mask_win.shape[1]))),
                                          axis=1)
                vad_win = np.concatenate((vad_win, np.zeros((self.win_len - vad_win.shape[1]))),
                                         axis=1)
                in_wins.append(in_win.T)
                mask_wins.append(mask_win.T)
                vad_wins.append(vad_win.T)
                n += self.win_hop
            else:
                k += 1
                n = 0
                is_loaded = 0

        return in_wins, mask_wins, vad_wins

    def __getitem__(self, item):
        if not self.test_set:
            in_stft, gt_mask, gt_vad = self.get_train_batch(item)
        else:
            in_stft, gt_mask, gt_vad = self.get_test_batch(item)

        return np.array(in_stft), [np.array(gt_mask), np.array(gt_vad)]


class RandomSequence(SubWinSequence):
    def __init__(self, *args, **kwargs):
        SubWinSequence.__init__(self, *args, **kwargs)
        a, b = SubWinSequence.__getitem__(self, 0)
        self.input_shape, self.output_shape = a.shape, b.shape

    def __getitem__(self, idx):
        return np.random.standard_normal(self.input_shape), np.random.standard_normal(self.output_shape)
