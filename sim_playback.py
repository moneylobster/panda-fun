'''
play back a recording made in the simulator
'''
import sys

import numpy as np
import roboticstoolbox as rtb
from swift import Swift

def load_and_playback(logname, stride):
    log=np.load(logname, allow_pickle=True)
    log=log.item()
    # print(log.keys())
    qs=np.array(log["q"])
    playback(qs, stride)

def playback(qs,stride):
    # create sim env
    env=Swift()
    env.launch(realtime=True)
    panda=rtb.models.Panda()
    env.add(panda)
    dt=0.001*stride
    for q in qs[::stride]:
        panda.q=q
        env.step(dt)
    
if __name__=="__main__":
    if len(sys.argv)<2:
        raise RuntimeError("Please supply the name of the .npy file to play.")
    logname=sys.argv[1]
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 50
    playback(logname, stride)
