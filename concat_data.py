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
sok_names=glob("fess_solutions/*")

for sok_name in sok_names:
    txt=int(sok_name[-3:])
    #read file
    with open(f"{sok_name}\\solutions.sok", "r") as f:
        sok=f.read()
    # go through all the levels solved in the file
    lvl=1
    solutionindex=0
    while True:
        lvlindex=sok.find(f"Level {lvl}")
        if lvlindex==-1:
            # end of levels
            break
        solutionindex=sok.find("Solution", lvlindex)
        if solutionindex==-1:
            # ended on unsolved level
            break
        #get action text
        actstartindex=sok.find("\n",solutionindex)+1
        actstopindex=sok.find("\n",actstartindex)
        acttext=sok[actstartindex:actstopindex]
        # play out the level to get the obs and act data
        obs=[]
        act=[]
        actindex=0
        s=Sokoban()
        s.loadlevel("medium","train",txt,lvl-1)
        while True:
            cmd=acttext[actindex].lower()
            if cmd=="u":
                move=0
            elif cmd=="r":
                move=1
            elif cmd=="d":
                move=2
            elif cmd=="l":
                move=3
            else:
                print(f"WARNING: encountered {cmd} in solution to {lvl}!")

            obs.append(s.level.flatten())
            #act.append(s.move) # save actions (akin to velocity control)
            s.move(move)
            act.append(s.playerpos) # save player position

            if s.checkwin():
                print(s.render())
                print(f"Level {lvl} completed.")
                obs.append(s.level.flatten())
                act.append(s.playerpos)
                break

            actindex+=1
            if actindex>=len(acttext):
                print(f"WARNING: the solution for {lvl} didn't finish it.")
                break

        obs=np.array(obs)
        act=np.array(act)
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

        # increment lvl
        lvl+=1

# # incrementally, if you find pairs, add them to the zarr file
# for obs_name in obs_names:
#     levelinfo=obs_name[5:12]
#     # check if it exists in act_names also
#     if f"data\\{levelinfo}_act.npy" in act_names:
#         obs=np.load(obs_name)
#         act=np.load(f"data\\{levelinfo}_act.npy")

#         # check if this is the first time adding to the array
#         # this is not a great approach but whatever
#         if "obsdata" in globals() and "actdata" in globals():
#             obsdata.append(obs)
#             actdata.append(act)
#             enddata.append(np.array([len(obs)]))
#         else:
#             obsdata=zarr.array(obs, dtype="float32")
#             actdata=zarr.array(act, dtype="float32")
#             enddata=zarr.array(np.array([len(obs)]))
#     else:
#         print(f"WARNING: {levelinfo} was in obs but not in act!")

# format zarr properly
store=zarr.ZipStore("sokobanfass.zarr.zip")
alldata=zarr.group(store=store, overwrite=True)
alldata.create_group("data")
alldata.create_group("meta")

alldata.data.create_dataset("state", data=obsdata)
alldata.data.create_dataset("action", data=actdata)
alldata.meta.create_dataset("episode_ends", data=enddata)

store.close()

