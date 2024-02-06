'''
concatenates all the data in the data folder to a .zarr file

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

# episode_ends should give the index of the first of the next episode.

# get the names of files
obs_names=glob("data/*_obs.npy")
act_names=glob("data/*_act.npy")

# incrementally, if you find pairs, add them to the zarr file
for obs_name in obs_names:
    levelinfo=obs_name[5:12]
    # check if it exists in act_names also
    if f"data\\{levelinfo}_act.npy" in act_names:
        obs=np.load(obs_name)
        act=np.load(f"data\\{levelinfo}_act.npy")

        # check if this is the first time adding to the array
        # this is not a great approach but whatever
        if "obsdata" in globals() and "actdata" in globals():
            obsdata.append(obs)
            actdata.append(act)
            enddata.append(np.array([len(obs)]))
        else:
            obsdata=zarr.array(obs, dtype="float32")
            actdata=zarr.array(act, dtype="float32")
            enddata=zarr.array(np.array([len(obs)]))
    else:
        print(f"WARNING: {levelinfo} was in obs but not in act!")

# format zarr properly
store=zarr.ZipStore("sokoban.zarr.zip")
alldata=zarr.group(store=store, overwrite=True)
alldata.create_group("data")
alldata.create_group("meta")

alldata.data.create_dataset("state", data=obsdata)
alldata.data.create_dataset("action", data=actdata)
alldata.meta.create_dataset("episode_ends", data=enddata)

store.close()
