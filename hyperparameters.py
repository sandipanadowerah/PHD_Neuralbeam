import glob
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




noises = ['robovox']  # , 'SSN', 'tv', 'interTalker'] in our case free-sound
nb_rir_train = 10000
nb_rir_test=1000
fs = 16000
#%% global parameters
# Spectral properties
n_fft = 512
hop_length = 256
# Room geometry
len_min = 3
len_max = 8
wid_min = 3
wid_max = 5
hei_min = 2
hei_max = 3
# Room acoustics
beta_min = 0.3
beta_max = 0.6
# Sensor positions
d = 1.5           # Array length
d_wal = 1       # Minimal distance to the walls
d_mic = 0.1     # Distance between mics and node centre
z_cst = 1.5       # Height of the array
d_rnd_mics = 1      # Distance between the two random mics
# Source properties
r_sou = 2.5     # Constant distance of sources to array centre
d_sou = 1.50    # Minimal distance of sources to microphones
d_sou_wal = 0.25
alpha_sou_min = 25
alpha_sou_max = 90
snr_lp = 0  # SNR at loudspeakers
# Source signal properties
dur_min_test = 1
dur_min_train = 5
dur_max = 10
max_order = 20
var_max = db2lin(-20)
snr_min, snr_max = 0, 10
beta_range = [0.3, 0.4, 0.5, 0.6 ]
early_reverb_len = 800 # 100ms
phi_range = [25,30,45,60,90]
n_mics = 3
