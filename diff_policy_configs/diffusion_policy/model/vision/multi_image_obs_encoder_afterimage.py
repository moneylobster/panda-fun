from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
import numpy as np

# A ImagePolicy takes observation dictionary:
#     "key0": Tensor of shape (B,To,*)
#     "key1": Tensor of shape e.g. (B,To,H,W,3) ([0,1] float32)
# and predicts action dictionary:
#     "action": Tensor of shape (B,Ta,Da)
# A ImageDataset returns a sample of dictionary:
#     "obs": Dict of
#         "key0": Tensor of shape (To, *)
#         "key1": Tensor fo shape (To,H,W,3)
#     "action": Tensor of shape (Ta, Da)
# Its get_normalizer method returns a LinearNormalizer with keys "key0","key1","action".

class AfterimageGenerator():
    def __init__(self, n_obs_steps, schedule_type):
        self.schedule = self.create_schedule(n_obs_steps, schedule_type)
        
    def create_schedule(self, n_obs_steps, schedule_type):
        """Create a schedule. n_obs_steps is the obs. horizon.
        Schedule types:
        - linear
        """
        if schedule_type=="linear":
            return torch.linspace(0,1,n_obs_steps)
        else:
            raise NotImplementedError(f"Unsupported schedule type {schedule_type}")
    
    def forward(self, images):
        """Create an afterimage from images according to weight schedule."""
        # basically a weighted avg
        return (self.schedule.view(self.schedule.shape[0],1,1,1) * images).sum(dim=0)/self.schedule.sum() # way slower than the np version idk why
        # return np.average(images, axis=0, weights=self.schedule)
    
def test_create_afterimage():
    import skimage.io as io
    import numpy as np
    imgs=[io.imread("brazil_renovated.jpg")]
    for i in range(1,10):
        imgs.append(np.roll(imgs[0],i*10,1))
    print(np.array(imgs).shape)
    ag=AfterimageGenerator(10, "linear")
    res=ag.forward(torch.Tensor(imgs))
    print(res.shape)
    return res
    

class MultiImageObsEncoderAfterimage(MultiImageObsEncoder):
    """Everything is the same as MultiImageObsEncoder, except to
    generate the encoding, we do not pass each image from the model,
    instead we compute an afterimage and pass that through.
    """

    def __init__(self,
                 shape_meta: dict,
                 rgb_model: Union[nn.Module, Dict[str,nn.Module]],
                 n_obs_steps,
                 resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
                 crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
                 random_crop: bool=True,
                 # replace BatchNorm with GroupNorm
                 use_group_norm: bool=False,
                 # use single rgb model for all rgb inputs
                 share_rgb_model: bool=False,
                 # renormalize rgb input with imagenet normalization
                 # assuming input in [0,1]
                 imagenet_norm: bool=False
                 ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.ke
    
    def forward(self, obs_dict):
        print("DEBUGG")
        print(f"Obsdict: {obs_dict}")
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                print(f"imgshape {img.shape}") # yields torch.Size([1, 3, 240, 320])
                # or 128, 3, 240, 320
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
                print(f"lastimgshape {img.shape}") # yields torch.Size([1, 3, 216, 288])
                print(f"featshape {feature.shape}") # yields torch.Size([1, 512]) or 128,512
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
