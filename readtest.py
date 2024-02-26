'''
just some tests reading and looking at recordings
'''
from glob import glob
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb
from skimage import io
from swift import Swift

SIM=True
T=0.1

firstact=glob("data/*_act.npy")[0]
firstobs=firstact[:-7]+"obs.npy"

print(f"reading {firstobs}")

imgsinv=np.load(firstobs)
# imgs=imgsinv[:,:,:,::-1]

log=np.load(firstact, allow_pickle=True)
log=log.item()
# log.keys():
# dict_keys(['K_F_ext_hat_K', 'O_F_ext_hat_K', 'O_T_EE', 'control_command_success_rate', 'dq', 'elbow', 'q', 'tau_J', 'tau_ext_hat_filtered', 'time'])
qs1khz=log["q"]

print(f"LEN ACT {len(qs1khz)} LEN OBS {len(imgsinv)}")
print("Truncating...")
truncto=floor(min(len(imgsinv)*1000*T, len(qs1khz))/(T*1000))
imgs=imgsinv[:int(truncto),:,:,::-1]
qs=qs1khz[:int(truncto*1000*T):int(1000*T)]
# if len(imgs)*100>len(qs1khz):
#     print(f"imglen {len(imgs)*100} is larger than actlen {len(qs1khz)}")
    
# else:
#     qs=qs1khz[:len(imgs)*1000/T:1000/T]

print(f"LEN ACT {len(qs)} LEN OBS {len(imgs)}")

# create sim env
if SIM:
    env=Swift()
    env.launch(realtime=True)
    panda=rtb.models.Panda()
    env.add(panda)

plt.ion()
if SIM:
    for i, img in enumerate(imgs):
        panda.q=qs[i]
        env.step(T)
        io.imshow(img)
        plt.pause(0.05)
else:
    for i, img in enumerate(imgs):
        io.imshow(img)
        plt.pause(0.05)
