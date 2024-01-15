#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define different architectures I might come to use
"""
import numpy as np
from keras import Sequential
from keras.layers import Bidirectional, CuDNNGRU, GRU, TimeDistributed, Dense, LSTM, Conv2D, Conv2DTranspose
from keras.layers import Input, Reshape, Dropout, MaxPool2D, BatchNormalization, ZeroPadding2D, LeakyReLU, concatenate
from keras.layers import Cropping2D
from keras.initializers import RandomUniform
from keras.models import Model
import keras.backend as K
from keras.optimizers import RMSprop
from code_utils.ml_utils.misc import extract_frames
from code_utils.ml_utils.my_losses import reconstruction_mse


def biGru3d(input_shape, output_dim):

    # check if we are on a GPU
    on_gpu = len(K.tensorflow_backend._get_available_gpus()) > 0

    gru = Sequential()
    if on_gpu:
        gru.add(Bidirectional(CuDNNGRU(units=input_shape[1],
                                       return_sequences=True),	    # Other arguments are not tunable
                              input_shape=input_shape))		    # the same as in standard GRU

    else:
        gru.add(Bidirectional(GRU(units=input_shape[1],
                                  activation='tanh',
                                  dropout=0.5,
                                  return_sequences=True,
                                  recurrent_activation='sigmoid',         # For original GRU 1406.1078v1 (see keras.io)
                                  reset_after=True),                      # Compatible with CuDNNGRU
                              input_shape=input_shape))

    gru.add(Dropout(0.5))
    gru.add(TimeDistributed(Dense(units=513, activation='relu')))
    gru.add(Dropout(0.5))
    gru.add(TimeDistributed(Dense(units=513, activation='relu')))
    gru.add(Dropout(0.5))
    gru.add(TimeDistributed(Dense(units=output_dim, activation='sigmoid')))

    return gru


def gru1d3(input_shape, units_nb, output_dim, dropout_frac=0.5, ker_reg=None):
    """
    Monodirectional GRU followed by 3 fully-connected layers. Dropout ratio of 0.5 for the 2 first FC.
    """

    gru = Sequential()
    gru.add(GRU(units=units_nb,
                activation='tanh',
                dropout=dropout_frac,
                kernel_regularizer=ker_reg,
                recurrent_regularizer=None,
                return_sequences=True,
                recurrent_activation='sigmoid',  # For original GRU 1406.1078v1 (see keras.io)
                reset_after=True,  # Compatible with CuDNNGRU
                input_shape=input_shape))

    gru.add(TimeDistributed(Dense(units=513, activation='relu')))
    gru.add(Dropout(0.5))
    gru.add(TimeDistributed(Dense(units=513, activation='relu')))
    gru.add(Dropout(0.5))
    gru.add(TimeDistributed(Dense(units=output_dim, activation='sigmoid')))

    return gru


def gru1d1(input_shape, units_nb, output_dim, dropout_frac=0.5, return_seq=True, ker_reg=None):
    """
    Monodirectional GRU followed by 1 fully-connected layer.
    """

    # Model structure
    input_layer = Input(shape=input_shape)
    x = GRU(units=units_nb,
            activation='tanh',
            dropout=dropout_frac,
            kernel_regularizer=ker_reg,
            recurrent_regularizer=None,
            return_sequences=return_seq,
            recurrent_activation='sigmoid',  # For original GRU 1406.1078v1 (see keras.io)
            reset_after=True,  # Compatible with CuDNNGRU
            input_shape=input_shape)(input_layer)
    output_layer = Dense(units=output_dim, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def c_gru1d1(input_shape, filters, kernel_sizes, c_strides, pool_sizes, rnn_units, output_dim, dropout=0.5,
             return_seq=True):
    second_dim = input_shape[1]
    input_layer = Input(shape=input_shape)
    x = input_layer
    for i in range(len(filters)):
        x = Conv2D(filters=filters[i],
                   kernel_size=kernel_sizes[i],  # Kernel size
                   strides=c_strides[i],            # Strides in y, x directions
                   padding="same",                  # If 'same', input is padded so that output shape equal to input one
                                                    # Else 'valid'
                   data_format='channels_last',     # see ~/.keras/keras.json
                   activation=None,
                   use_bias=True
                   )(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPool2D(pool_size=pool_sizes[i],
                      strides=None,
                      padding='valid')(x)
        second_dim = int(second_dim / pool_sizes[i][1])
    x = Reshape((-1, second_dim * filters[-1]))(x)
    x = GRU(units=rnn_units,
            activation='tanh',
            dropout=dropout,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            return_sequences=return_seq,
            recurrent_activation='sigmoid',  # For original GRU 1406.1078v1 (see keras.io)
            reset_after=True)(x)
    output_layer = Dense(units=output_dim, activation='sigmoid')(x)

    cnv_gru = Model(inputs=input_layer, outputs=output_layer)

    return cnv_gru


def gru1d1_mtl(input_shape, units_nb, out_dims, dropout_frac=0.5, return_seq=True, ker_reg=None):
    """
    gru1d1 with two parallel FF on last layer for multi task learning (MLT)
    """
    input_layer = Input(shape=input_shape)
    x = GRU(units=units_nb,
            activation='tanh',
            dropout=dropout_frac,
            kernel_regularizer=ker_reg,
            recurrent_regularizer=None,
            return_sequences=return_seq,
            recurrent_activation='sigmoid',  # For original GRU 1406.1078v1 (see keras.io)
            reset_after=True)(input_layer)
    out_mask = Dense(units=out_dims[0], activation='sigmoid')(x)
    out_vad = Dense(units=out_dims[1], activation='sigmoid')(x)

    gru = Model(inputs=input_layer, outputs=[out_mask, out_vad])

    return gru


class CNN:
    def __init__(self, input_shape,
                 filters, kernel_sizes, c_strides, pool_sizes,
                 c_padding='same', dilation_rate=(1, 1), activation_function=None,
                 batch_normalization_axis=1,
                 pool_strides=None, pool_padding='valid'):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.c_strides = c_strides
        self.pool_sizes = pool_sizes

        nb_filters = len(filters)
        if type(c_padding) != list:
            self.c_padding = np.tile(c_padding, nb_filters)
        else:
            self.c_padding = c_padding
        if np.shape(dilation_rate)[0] != nb_filters:
            self.dilation_rate = np.tile(dilation_rate, nb_filters).reshape((nb_filters, -1))
        else:
            self.dilation_rate = dilation_rate
        if type(activation_function) != list:
            self.activation = np.tile(activation_function, nb_filters)
        else:
            self.activation = activation_function
        if type(batch_normalization_axis) != list:
            self.batch_norm_axis = np.tile(batch_normalization_axis, nb_filters)
        else:
            self.batch_norm_axis = batch_normalization_axis
        if type(pool_strides) != list:
            self.pool_strides = np.tile(pool_strides, nb_filters)
        else:
            self.pool_strides = pool_strides
        if type(pool_padding) != list:
            self.pool_padding = np.tile(pool_padding, nb_filters)
        else:
            self.pool_padding = pool_padding

    def architecture(self):
        feature_dim = int(self.input_shape[1])
        input_layer = Input(shape=self.input_shape)
        x = input_layer
        for i in range(len(self.filters)):
            if self.c_padding[i] == 'valid':
                # We decide to keep the feature dimension constant after convolution
                # Even with 'valid' (change only the time dimension)
                n = feature_dim - int(np.floor((feature_dim - self.kernel_sizes[i][1]) / self.c_strides[i][1]) + 1)
                x = ZeroPadding2D(padding=((0, 0), (int(np.floor(n / 2)), int(np.ceil(n / 2)))))(x)
            x = Conv2D(filters=self.filters[i],
                       kernel_size=self.kernel_sizes[i],  # Kernel size
                       strides=self.c_strides[i],            # Strides in y, x directions
                       padding=self.c_padding[i],                   # If 'same', input is padded so that
                                                                    # output shape equal to input one. # Else 'valid'
                       data_format='channels_last',     # see ~/.keras/keras.json
                       activation=self.activation[i],
                       use_bias=True,
                       dilation_rate=self.dilation_rate[i]
                       )(x)
            if self.batch_norm_axis[i] is not None:
                x = BatchNormalization(axis=self.batch_norm_axis[i])(x)
            if self.pool_sizes[i] is not None:
                x = MaxPool2D(pool_size=self.pool_sizes[i],
                              strides=self.pool_strides[i],
                              padding=self.pool_padding[i])(x)
                feature_dim = int(np.floor(feature_dim / self.pool_sizes[i][1]))
        output_layer = Reshape((-1, feature_dim * self.filters[-1]))(x)

        return input_layer, output_layer

    def build_model(self):
        self.input_layer, self.output_layer = self.architecture()
        return Model(inputs=self.input_layer, outputs=self.output_layer)


class RNN:
    def __init__(self, input_shape, hidden_layers_units, cell_architecture, cell_activations,
                 dropouts, bidirectional_layers=0, kernel_regularizer=None, return_sequences=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'):
        self.input_shape = input_shape
        self.hidden_units = hidden_layers_units
        if type(bidirectional_layers) == int:
            self.bi_wrappers = bidirectional_layers * np.ones(len(hidden_layers_units))
        else:
            self.bi_wrappers = bidirectional_layers
        self.cell_archi = cell_architecture
        if len(np.shape(cell_activations)) == 1:
            self.cell_activs = np.tile(cell_activations, len(hidden_layers_units)).reshape((len(hidden_layers_units), -1))
        else:
            self.cell_activs = cell_activations
        self.dropouts = dropouts
        self.ker_reg = kernel_regularizer
        self.return_sequences = return_sequences
        self.ker_init = kernel_initializer
        self.rec_init = recurrent_initializer

    def architecture(self, input_layer=None):
        if str.lower(self.cell_archi) == 'lstm':
            rnn_layer = LSTM
        elif str.lower(self.cell_archi) == 'gru':
            rnn_layer = GRU
        else:
            raise ValueError('LSTM and GRU layers can be implemented so far')

        if input_layer is None:
            input_layer = Input(self.input_shape)
        x = input_layer
        for i in range(len(self.hidden_units)):
            if self.bi_wrappers[i] == 1:
                if i == len(self.hidden_units) - 1:
                    x = Bidirectional(rnn_layer(
                        units=self.hidden_units[i],
                        activation=self.cell_activs[i][0],
                        dropout=self.dropouts[i],
                        kernel_regularizer=self.ker_reg,
                        return_sequences=self.return_sequences,
                        recurrent_activation=self.cell_activs[i][1],
                        kernel_initializer=self.ker_init,
                        recurrent_initializer=self.rec_init
                    ), merge_mode='ave')(x)
                    output_layer = x
                else:
                    x = Bidirectional(rnn_layer(
                        units=self.hidden_units[i],
                        activation=self.cell_activs[i][0],
                        dropout=self.dropouts[i],
                        kernel_regularizer=self.ker_reg,
                        return_sequences=True,
                        recurrent_activation=self.cell_activs[i][1]
                    ), merge_mode='concat')(x)
            else:
                if i == len(self.hidden_units) - 1:
                    x = rnn_layer(
                        units=self.hidden_units[i],
                        activation=self.cell_activs[i][0],
                        dropout=self.dropouts[i],
                        kernel_regularizer=self.ker_reg,
                        return_sequences=self.return_sequences,
                        recurrent_activation=self.cell_activs[i][1]
                    )(x)
                    output_layer = x
                else:
                    x = rnn_layer(
                        units=self.hidden_units[i],
                        activation=self.cell_activs[i][0],
                        dropout=self.dropouts[i],
                        kernel_regularizer=self.ker_reg,
                        return_sequences=True,
                        recurrent_activation=self.cell_activs[i][1]
                    )(x)
        return input_layer, output_layer

    def build_model(self):
        self.input_layer, self.output_layer = self.architecture()
        return Model(inputs=self.input_layer, outputs=self.output_layer)


class FF:
    def __init__(self, input_shape, nb_units, activations):
        self.input_shape = input_shape
        self.units = nb_units
        nu = len(nb_units)
        if len(np.shape(activations)) == 1:
            self.activations = np.tile(activations, nu).reshape((nu, -1))
        else:
            self.activations = activations
        self.activations = activations

    def architecture(self, input_layer=None):
        if input_layer is None:
            input_layer = Input(self.input_shape)

        x = input_layer
        for i in range(len(self.units)):
            x = Dense(units=self.units[i], activation=self.activations[i])(x)

        return input_layer, x

    def build_model(self):
        self.input_layer, self.output_layer = self.architecture()
        return Model(inputs=self.input_layer, outputs=self.output_layer)


class CRNN(CNN):
    def __init__(self, input_shape,
                 filters, kernel_sizes, c_strides, pool_sizes,
                 rnn_hidden_layers, rnn_cells, rnn_activations,
                 rnn_dropouts, bidirectional_layers=0, kernel_regularizer=None, return_sequences=True,
                 c_padding='same', dilation_rate=(1, 1), activation_function=None,
                 batch_normalization_axis=1,
                 pool_strides=None, pool_padding='valid',
                 ff_units=None, ff_activation=None):
        CNN.__init__(self, input_shape,
                     filters, kernel_sizes, c_strides, pool_sizes, c_padding, dilation_rate, activation_function,
                     batch_normalization_axis,
                     pool_strides, pool_padding)

        self.hidden_units = rnn_hidden_layers
        self.bi_wrappers = bidirectional_layers
        self.cell_archi = rnn_cells
        self.cell_activs = rnn_activations
        self.dropouts = rnn_dropouts
        self.ker_reg = kernel_regularizer
        self.return_sequences = return_sequences
        # FF structure
        self.ff_units = ff_units
        self.ff_activations = ff_activation

    def architecture(self):
        input_layer, cnn_layer = CNN.architecture(self)
        rnn_class = RNN(input_shape=None, hidden_layers_units=self.hidden_units,
                        bidirectional_layers=self.bi_wrappers,
                        cell_architecture=self.cell_archi, cell_activations=self.cell_activs, dropouts=self.dropouts,
                        kernel_regularizer=None, return_sequences=self.return_sequences)
        _, output_layer = rnn_class.architecture(input_layer=cnn_layer)
        if self.ff_units is not None:
            ff_class = FF(input_shape=None, nb_units=self.ff_units, activations=self.ff_activations)
            _, output_layer = ff_class.architecture(input_layer=output_layer)

        return input_layer, output_layer


class Heymann:
    """
    Construct the BiLSTM of [1].
    This class has the particularity to also have a compiling function as the optimizer is imposed by the reference.

    [1] Jahn Heymann, Lukas Drude, and Reinhold Haeb-Umbach,
        “Neural network based spectral mask esti-mation for acoustic beamforming,”
        in ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing -
        Proceedings, 2016, vol. 2016-May, pp.196–200.
    [2] https://github.com/fgnt/nn-gev/blob/master/fgnt/chainer_extensions/links/sequence_lstms.py
    """
    def __init__(self, input_shape, n_fft=512):
        """
        :param input_shape:         Shape of the input (batch_size x n_time x n_freq)
        :param n_fft                FFT length (important if input is stacked features -> loss will be computed on lower feature)
        """
        self.input_shape = input_shape
        self.rnn_units = 256
        self.rnn_dropout = 0.5
        self.rnn_init_min = -0.04
        self.rnn_init_max = 0.04
        self.weight_initializer = RandomUniform(minval=self.rnn_init_min, maxval=self.rnn_init_max)
        self.ff12_units = 513
        self.ff12_activation = 'relu'
        self.ff3_units = 257    # 513 in [1]
        self.ff3_activation = 'sigmoid'
        self.ff_dropout = 0.5
        # Optimizer
        self.lr = 0.001
        self.heymann = self.build_model()
        n_features = int(np.floor(n_fft / 2) + 1)
        self.loss = reconstruction_mse(extract_frames(self.heymann.input[:, :, :n_features],
                                                      'mid', three_d_tensor=False))
        self.metrics = ['mse']

    def build_model(self):
        input_layer = Input(self.input_shape)
        # BiLSTM layer
        x = Bidirectional(LSTM(units=self.rnn_units,
                               activation='tanh',
                               dropout=self.rnn_dropout,   # Equivalent of line 69 in [2]
                               recurrent_dropout=0,     # No recurrent dropout according to code
                               kernel_regularizer=None,
                               return_sequences=False,
                               recurrent_activation='sigmoid',
                               kernel_initializer=self.weight_initializer,
                               recurrent_initializer=self.weight_initializer),
                          merge_mode='concat')(input_layer)
        x = BatchNormalization()(x)         # Should be applied right after dropout in LSTM but no access here
        # FF 1 + Dropout
        x = Dropout(rate=self.ff_dropout)(x)
        x = Dense(units=self.ff12_units,
                  activation=self.ff12_activation,
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        # FF2 + Dropout
        x = Dropout(rate=self.ff_dropout)(x)
        x = Dense(units=self.ff12_units,
                  activation=self.ff12_activation,
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        # FF3 No dropout
        output_layer = Dense(units=self.ff3_units,
                             activation=self.ff3_activation,
                             kernel_initializer='glorot_uniform')(x)

        heymann = Model(inputs=input_layer, outputs=output_layer)

        return heymann

    def compile_model(self):
        # Clip gradient
        model_optimizer = RMSprop(lr=self.lr, clipnorm=1.)
        self.heymann.compile(optimizer=model_optimizer,
                             loss=self.loss,
                             metrics=self.metrics)


class Jansson():
    """ Class to create AudioUnet described in [1]
    # Arguments
        n_frames: number of frames of the input spectrogram
        n_freq: number of frequency bins in the input spectrogram
        
        [1] A. Jansson, E. J. Humphrey, N. Montecchio, R. M. Bittner, A. Kumar,and T. Weyde,
            “Singing voice separation with deep u-net convolutionalnetworks,” inProc. of ISMIR, 2017
    """

    def __init__(self, input_shape):
        """ Constructor: See doc of the class """
        self.format = K.image_data_format()
        self.frames = input_shape[0]
        self.freq = input_shape[1]

        if self.format == 'channels_first':
            self.ch_axis = 1
            self.input_shape = (1, self.frames, self.freq)
        elif self.format == 'channels_last':
            self.ch_axis = 3
            self.input_shape = (self.frames, self.freq, 1)
        self.norm_axis = self.input_shape.index(self.freq) + 1      # +1 because of Batch dimension

        # Contraction
        self.n_filters_c = [16, 32, 64, 128, 256, 512]
        self.kernels_c = [(5, 5), (5, 5), (3, 3), (5, 5), (5, 5), (5, 5)]
        self.strides_c = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        # Expansion
        self.n_filters_e = [256, 128, 64, 32, 16, 1]
        self.kernels_e = [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
        self.strides_e = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]

        self.jansson = self.build_model()

    def build_model(self):
        """ Build AudioUnet
        # Arguments
            optimizer: A Keras optimizer with which to compile the model
        # Returns
            A compiled Keras Model
        We use the reconstruction loss here, which is the mean squared error on
        the mask applied to the input spectrogram
        """

        encoded = []
        inp = Input(shape=self.input_shape)
        iinp = Cropping2D(cropping=((0, 0), (1, 0)))(inp)
        enc = iinp
        for f, k, s in zip(self.n_filters_c, self.kernels_c, self.strides_c):
            x = Conv2D(f, k, data_format=self.format, activation=None, strides=s, padding='same')(enc)
            x = BatchNormalization(axis=self.norm_axis)(x)
            encoded.append(x)
            enc = LeakyReLU(alpha=0.2)(x)

        encoded.pop()
        dec = x

        i_dconv = 0
        for f, k, s in zip(self.n_filters_e[:-1], self.kernels_e[:-1], self.strides_e[:-1]):
            x = Conv2DTranspose(f, k, strides=s, padding='same',
                                data_format=self.format, activation='relu')(dec)
            if i_dconv < 3:
                x = Dropout(rate=0.5)(x)
                i_dconv += 1
            x = BatchNormalization(axis=self.norm_axis)(x)
            to_concat = encoded.pop()
            dec = concatenate([x, to_concat], axis=self.ch_axis)

        out = Conv2DTranspose(self.n_filters_e[-1], self.kernels_e[-1], strides=self.strides_e[-1],
                              data_format=self.format, padding='same', activation='sigmoid')(dec)
        out = ZeroPadding2D(((0, 0), (1, 0)))(out)  # Compensate cropping at the beginning
        out = Reshape((self.frames, self.freq))(out)
        self.model = Model(inputs=inp, outputs=out)
        return self.model

    def compile_model(self, optimizer='Adam'):
        """ Builds the model and compiles it with Adam optimizer
            optimizer: a Keras optimizer with which to compile the model
        # Returns
             A compiled Keras Model
        """
        unet = self.jansson
        self.jansson.compile(loss=reconstruction_mse(extract_frames(unet.input, 'all', three_d_tensor=True)),
                             optimizer=optimizer)
