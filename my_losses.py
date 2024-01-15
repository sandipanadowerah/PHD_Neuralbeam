"""
Self-created loss functions
"""
from keras import backend as K
from keras.losses import mean_squared_error


def mse_l1l2(l1, l2):
    """
    L1-L2 regularized mean squared error loss function:
    loss(y_true, y_pred) = mean_square(y_pred - y_true) + l1*mean_absolute(y_pred) + l2*mean_square(y_pred)
    :param l1:          Regularization factor before norm 1 term
    :param l2:          Regularization factor before norm 2 term
    :return:            loss *function*
    """
    def loss_fun(y_true, y_pred):
        mse_term = mean_squared_error(y_true, y_pred)
        l1l2_term = l1*K.mean(K.abs(y_pred), axis=-1) + l2*K.mean(K.square(y_pred), axis=-1)
        return mse_term + l1l2_term
    return loss_fun


def my_kl():
    """
    Kullbak-Leibler divergence with regularization term. As this is supposed to be used as loss function, ergo to be 
    minimized, the absolute value of the divergence is returned
    """
    def loss_fun(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        kl_reg = K.sum(y_true * K.log(y_true / y_pred), axis=-1) + (y_pred - y_true)
        return abs(kl_reg)
    
    return loss_fun


def reconstruction_mse(stft):
    """ MSE on mask applied on input STFT.
    :param stft:    tensor of same shape as y_true and y_pred, frame(s) of
                    the input spectrograms to apply the mask on.
    :return:        A Keras loss function, the mse weighted by the input stft.
    """
    def loss(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true) * stft), axis=-1)
    return loss


def jensen_shannon(stft):
    """
    Jensen-Shannon divergence between ground truth mask applied on input and predicted mask applied on input.
    :param stft:    tensor of same shape as y_true and y_pred, frame(s) of
                    the input spectrograms to apply the mask on.
    :return:        A Keras loss function, equal to the Jensen-Shannon divergence between ground truth and predicted
                    mask applied on input mixture STFT.
    """
    def loss(y_true, y_pred):
        x = K.clip(y_true * stft, K.epsilon(), 1e6)
        y = K.clip(y_pred * stft, K.epsilon(), 1e6)
        return K.mean(1 / 2 * (x * K.log(2 * x / (x + y)) + y * K.log(2 * y / (x + y))))
    return loss
