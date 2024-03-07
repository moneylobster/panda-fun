'''
Processes and concatenates all the data in the data folder to a .zarr file

the structure from the pusht example is:
/
 ├── data
 │   ├── action (25650, 2) float32
 │   ├── img (25650, 96, 96, 3) float32
 │   ├── keypoint (25650, 9, 2) float32
 │   ├── n_contacts (25650, 1) float32
 │   └── state (25650, 5) float32
 └── meta
     └── episode_ends (206,) int64
'''
import zarr
from glob import glob
import numpy as np
from skill_utils.truncate import truncate
from skill_utils.format_pose import to_format

T=0.1 # period in seconds

# episode_ends should give the index of the first of the next episode.

# get the names of files
obs_names=glob("data/*_obs.npy")
act_names=glob("data/*_act.npy")

# incrementally, if you find pairs, add them to the zarr file
for obs_name in obs_names:
    timeinfo=obs_name[5:-8]
    # check if it exists in act_names also
    if f"data/{timeinfo}_act.npy" in act_names:
        #load
        obs=np.load(obs_name)
        actlog=np.load(f"data/{timeinfo}_act.npy", allow_pickle=True).item()
        #get O_T_EE and convert into 9d format
        act=[to_format(frame) for frame in actlog["O_T_EE"]]
        # subsample from 1kHz to 10Hz and trim to match camera recording length
        # (assuming camera len is shorter than rec)
        obs, act = truncate(obs, act, T)

        # check if this is the first time adding to the array
        # this is not a great approach but whatever
        if "obsdata" in globals() and "actdata" in globals():
            obsdata.append(obs)
            actdata.append(act)
            enddata.append(np.array([len(obs)+enddata[-1]]))
        else:
            obsdata=zarr.array(obs, dtype="float32")
            actdata=zarr.array(act, dtype="float32")
            enddata=zarr.array(np.array([len(obs)]))
    else:
        print(f"WARNING: {timeinfo} was in obs but not in act!")

# also create a "state" dataset that's just the actions delayed by a timestep
states=np.zeros_like(actdata)
states[1:]=actdata[:-1]
statedata=zarr.array(states)

# format zarr properly
store=zarr.ZipStore("panda.zarr.zip")
alldata=zarr.group(store=store, overwrite=True)
alldata.create_group("data")
alldata.create_group("meta")

alldata.data.create_dataset("img", data=obsdata)
alldata.data.create_dataset("action", data=actdata)
alldata.data.create_dataset("state", data=statedata)
alldata.meta.create_dataset("episode_ends", data=enddata)

store.close()
