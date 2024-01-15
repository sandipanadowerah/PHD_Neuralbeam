#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import sys
import collections
import numpy as np
import soundfile as sf
import time

#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import soundfile as sf
from tqdm import tqdm

from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root

import json


# In[ ]:


from hyperparameters import *
from utils import *


# In[ ]:


launch_id = sys.argv[2]

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)


# In[26]:


data_path = config['data_path']#'/home/ajinkyak/workspace/projects/Voices_evaluation/fasnet_R1/enroll_eval'
dir_paths = config['dir_paths']#'/home/ajinkyak/workspace/projects/Voices_evaluation/fasnet_R1/enroll_eval_dir.txt'
s1_tag =    config['s1_tag']
step_size = config['step_size']

dir_list = [line.strip() for line in open(dir_paths, 'r')]

wav_scp = {}

for j, i in enumerate(range(0,len(dir_list), step_size)):
    wav_scp[j] = d[i:i+step_size]   


# In[25]:


for dir_ in wav_scp[launch_id]:
    for fname in os.listdir(os.path.join(data_path, dir_)):        if fname.find('s2_estimate') != -1:
            check_path = os.path.join(data_path, dir_, fname.replace('s2_estimate', '_fasnet_r1_wpe'))
            if os.path.exists(check_path) == False:
                s2,_ = sf.read(os.path.join(data_path, dir_, fname))
                s1,_ = sf.read(os.path.join(data_path, dir_, fname.replace('s2_estimate', 's1_estimate')))
                y,_ = sf.read(os.path.join(data_path, dir_, 'mixture.wav'))
            
                s2 = np.asanyarray([s2,s2,s2]).T
                s1 = np.asanyarray([s1,s1,s1]).T
                y = np.asanyarray([y,y,y]).T
            
                if s1_tag == 'noise':
                    s_stft = stft_nchannel_wave(s2)
                    n_stft = stft_nchannel_wave(s1)
                else:
                    n_stft = stft_nchannel_wave(s2)
                    s_stft = stft_nchannel_wave(s1)

                ms = ideal_ratio_mask(s_stft, n_stft)
                ms = np.transpose(ms, (2,0,1))

                f_mask_r1 = multichannel_weiner_filter_custom2(y, ms, mu=0.001)
                f_mask_r1_wpe = f_mask_r1.T # unet_ds_r1_wpe
                f_mask_r1_wpe = dereverb(f_mask_r1_wpe[:1,:])

                sf.write(os.path.join(data_path, dir_, fname.replace('s2_estimate', '_fasnet_r1')), f_mask_r1[:,0], sr)
                sf.write(os.path.join(data_path, dir_, fname.replace('s2_estimate', '_fasnet_r1_wpe')), f_mask_r1_wpe[:,0], sr)


# In[ ]:





# In[ ]:




