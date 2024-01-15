#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes gathering information used in training or testing
"""
from code_utils.ml_utils.my_losses import mse_l1l2, my_kl, reconstruction_mse, jensen_shannon
from keras.optimizers import RMSprop


class Conv2DConfiguration:
    def __init__(self, filters=1, kernel_size=(1, 1), strides=(1, 1), pool_sizes=(1, 1), c_padding='same'):
        self.c_filters = filters
        self.c_ker_size = kernel_size
        self.c_strides = strides
        self.pool_sizes = pool_sizes
        self.c_padding = c_padding


class RnnConfiguration:
    def __init__(self, cell_structure, units_per_layer, dropout_frac,
                 bi_dirs=0, activations=['tanh', 'sigmoid']):
        self.rnn_cell = cell_structure
        self.rnn_units = units_per_layer
        self.bidirs = bi_dirs
        self.rnn_dropout = dropout_frac
        self.rnn_activations = activations


class FfConfiguration:
    def __init__(self, units, ff_activations):
        self.ff_units = units
        self.ff_activations = ff_activations


class CrnnConfiguration:
    def __init__(self, architecture, conv_layer=None, rnn_layer=None, ff_layer=None, mtl=None):
        self.arch = architecture
        self.mtl = mtl
        self.conv_layer = conv_layer
        self.rnn_layer = rnn_layer
        self.ff_layer = ff_layer


class ModelConfiguration:
    """
    Some redundant parameters with CRNN are kept for compatibility purposes
    """
    def __init__(self, architecture, cell_structure, units_per_layer, dropout_frac,
                 bi_dirs=0, activations=['tanh', 'sigmoid'], mtl=None, conv_layer=None, ff_layer=None):
        self.arch = architecture
        self.rnn_cell = cell_structure
        self.rnn_units = units_per_layer
        self.bidirs = bi_dirs
        self.rnn_dropout = dropout_frac
        self.rnn_activations = activations
        self.mtl = mtl
        self.conv_layer = conv_layer
        self.ff_layer = ff_layer


class DataConfiguration:
    def __init__(self, train_noise, inter_noise, channel_nb, mask_type, z_folder='Noisy/z_donse_oracle/'):
        self.train_noise = train_noise
        self.inter_noise = inter_noise
        self.ch = channel_nb
        self.mtype = mask_type
        self.z_folder = z_folder


class TrainingConfiguration:
    def __init__(self, batch_size, epoch_nb, win_len, win_hop, norm="norm_on_seq", pred_frame='last', z_axis=0):
        self.batch_size = batch_size
        self.epoch_nb = epoch_nb
        self.win_len = win_len
        self.win_hop = win_hop
        self.norm = norm
        self.pred_frame = pred_frame
        self.z_axis = z_axis


class OptimizerConfiguration:
    def __init__(self, optimizer_algo=RMSprop, lr=0.001,
                 metric_list=['mean_squared_error', 'reconstruction_mse'],
                 loss_fcn=['mean_squared_error'], loss_params=[0, 0], frames_layer=None,
                 weights=[1]):
        self.optimizer = optimizer_algo
        self.lr = lr
        self.metrics = metric_list
        self.loss = loss_fcn
        self.loss_params = loss_params
        self.input_frames = frames_layer
        self.weights = weights

    def get_loss(self):
        loss_fn = []
        for loss_name in self.loss:
            if loss_name == 'mse':
                loss_fn.append("mean_squared_error")
            elif loss_name == 'bce':
                loss_fn.append("binary_crossentropy")
            elif loss_name == 'mse_l1l2':
                loss_fn.append(mse_l1l2(self.loss_params[0], self.loss_params[1]))
            elif loss_name in ['kullback_leibler_divergence', 'KLD', 'kld']:
                loss_fn.append(my_kl)
            elif loss_name in ['rec_mse', 'reconstruction_mse']:
                loss_fn.append(reconstruction_mse(self.input_frames))
            elif loss_name in ['jensen_shannon', 'js', 'JS']:
                loss_fn.append(jensen_shannon(self.input_frames))
            else:
                loss_fn.append(loss_name)

        return loss_fn

    def get_metrics(self):
        loss_fn = []
        for loss_name in self.metrics:
            if loss_name == 'mse':
                loss_fn.append("mean_squared_error")
            elif loss_name == 'bce':
                loss_fn.append("binary_crossentropy")
            elif loss_name == 'mse_l1l2':
                loss_fn.append(mse_l1l2(self.loss_params[0], self.loss_params[1]))
            elif loss_name in ['kullback_leibler_divergence', 'KLD', 'kld']:
                loss_fn.append(my_kl)
            elif loss_name in ['rec_mse', 'reconstruction_mse']:
                loss_fn.append(reconstruction_mse(self.input_frames))
            elif loss_name in ['jensen_shannon', 'js', 'JS']:
                loss_fn.append(jensen_shannon(self.input_frames))
            else:
                loss_fn.append(loss_name)

        return loss_fn
