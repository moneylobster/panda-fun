'''
just some tests reading and looking at recordings
'''
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb
from skimage import io
import cv2
from swift import Swift

from skill_utils.truncate import truncate

SIM=True
T=0.1

firstact=glob("data/*_act.npy")[1]
firstobs=firstact[:-7]+"obs.npy"

print(f"reading {firstobs}")

imgsinv=np.load(firstobs)

log=np.load(firstact, allow_pickle=True)
log=log.item()
# log.keys():
# dict_keys(['K_F_ext_hat_K', 'O_F_ext_hat_K', 'O_T_EE', 'control_command_success_rate', 'dq', 'elbow', 'q', 'tau_J', 'tau_ext_hat_filtered', 'time'])
qs1khz=log["q"]

print(f"LEN ACT {len(qs1khz)} LEN OBS {len(imgsinv)}")
print("Truncating...")
imgs=imgsinv[:,:,:,::-1]
imgs,qs=truncate(imgs, qs1khz, T)
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
        # cv2.imshow("Img",img)
        # cv2.waitKey(1)
else:
    for i, img in enumerate(imgs):
        io.imshow(img)
        plt.pause(0.05)
