'''
Test the model's responses to training data.
'''
import numpy as np
import cv2
import torch
import dill
import hydra
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from glob import glob
import os
import zarr

from skill_utils.truncate import truncate
from skill_utils.format_pose import to_format, from_format
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()


class Model():
    def __init__(self, ckpt_path, device="cuda"):
        print("Started initialization.")
        self.payload=torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        print("Loaded checkpoint.")
        self.cfg=self.payload["cfg"]
        self.cls=hydra.utils.get_class(self.cfg._target_)
        self.workspace=self.cls(self.cfg)
        print("Loading payload...")
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)
        
        print("Loading model...")
        if self.cfg.training.use_ema:
            self.policy=self.workspace.ema_model
        else:
            self.policy=self.workspace.model

        self.device=torch.device(device)
        self.policy.eval().to(self.device)
        # set inference params
        self.policy.num_inference_steps = 16 # DDIM inference iterations
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        self.obs_res=get_real_obs_resolution(self.cfg.task.shape_meta)
        print("Done.")

    def eval(self, obs):
        with torch.no_grad():
            self.policy.reset()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs, shape_meta=self.cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
            
            del result
        return action
            

class Dataset():
    """
    A dataset class to access data recorded using the default recorders
    Assumes data.action is 9D, data.robot_eef_pose is 16D
    """
    def __init__(self, zarr_path):
        with zarr.ZipStore(zarr_path, mode='r') as zip_store:
            self.dataset=ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
        self.action=self.dataset.data.action
        self.camera_0=self.dataset.data.camera_0
        self.robot_eef_pose=self.dataset.data.robot_eef_pose
        self.episode_ends=self.dataset.meta.episode_ends
    
def eval_at_index(model, dataset, index, horizon=2):
    "eval MODEL for observation at t=INDEX of DATASET using HORIZON steps"
    # format one of the observations into the dict format accepted by the model
    assert horizon-1<=index
    
    obsdict={
        "robot_eef_pose" : np.array(
            [dataset.robot_eef_pose[index-i] for i in range(0,horizon)]),
        "camera_0" : np.array(
            [dataset.camera_0[index-i] for i in range(0,horizon)]),
    }
    res=model.eval(obsdict)
    return res

def get_answers_at_index(dataset, index, horizon):
    "get the related data from the DATASET following INDEX for HORIZON steps."
    return np.array(
        [dataset.action[index+i] for i in range(0,horizon)])

def diff_at_index(model, dataset, index):
    "return difference of model prediction and dataset example"
    pred=eval_at_index(model, dataset, index, 8)
    ans=get_answers_at_index(dataset, index, 15)
    return pred-ans

# TODO
# def get_deltas_for_episode(model, episode):
#     "evaluate at all indices and compare with expected response"
#     results=[]
#     deltas=[]
#     for index in tqdm.tqdm(range(2,len(episode.obs))):
#         res=eval_at_index(model, episode, index)
#         results.append(res[0])
#         deltas.append(res[0]-to_format(episode.act[index]))
#     return results, deltas

# TODO
# def episode_delta_stats(model, episode):
#     results, deltas=get_deltas_for_episode(model,episode)
    
#     print(f"Min: {np.min(np.abs(deltas),0)}\nMax: {np.max(np.abs(deltas),0)}")

#     norms=np.array([np.linalg.norm(i) for i in deltas])
#     err_thres=0.3
#     errcnt=len(norms[norms<0.3])
#     print(f"Total err < {err_thres} count: {errcnt}")
#     return errcnt
    
# testing out what sort of response the model gives when prompted with training data
class Test():
    def __init__(self):
        # model
        outputs_dir="/home/romerur30/panda/diffusion_policy/data/outputs"
        train_day_dir="2024.08.02"
        train_hour="19.56.53"
        train_hour_dir=f"{train_hour}_train_diffusion_unet_image_real_push_image"
        checkpoint="latest.ckpt"
        # dataset
        rec_dir="/home/romerur30/panda/diffusion_policy/2d_data"
        dataset_name="6683f84cd8a4de38cc004bf59b3cc22f.zarr.zip"

        self.model=Model(os.path.join(outputs_dir,
                                 train_day_dir,train_hour_dir,
                                 "checkpoints",checkpoint))

        self.dataset=Dataset(os.path.join(rec_dir,dataset_name))

    def eval(self, index):
        return eval_at_index(self.model,self.dataset,index,8)
    
    def ans(self, index):
        return get_answers_at_index(self.dataset,index,15)

    def sidebyside(self,index):
        "plot eval and ans trajs from blue->red"
        colors = cm.rainbow(np.linspace(0, 1, 15))
        plt.subplot(1,2,1)
        plt.scatter(*(self.eval(index).T), c=colors)
        plt.subplot(1,2,2)
        plt.scatter(*(self.ans(index).T), c=colors)


t=Test()
t.sidebyside(1000)
