import numpy as np
import cv2
import torch
import dill
import hydra

from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class Model():
    def __init__(self, ckpt_path):
        self.payload=torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        self.cfg=payload["cfg"]
        self.cls=hydra.utils.get_class(cfg._target_)
        self.workspace=cls(cfg)
        self.workspace.load_payload(paylad, exclude_keys=None, include_keys=None)
        

    def eval():
        with torch.no_grad():
            pass


## INPUT
obs={}

## EVAL

## OUTPUT
