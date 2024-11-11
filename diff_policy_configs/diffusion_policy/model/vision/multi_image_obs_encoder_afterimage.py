from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

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
        self.n_obs_steps=n_obs_steps
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
    
    def afterimage(self, images):
        """Create an afterimage from images according to weight schedule.
        This needs the images to come in as multiples of n_obs_steps, if larger than it."""
        self.schedule=self.schedule.to(images) # should only take time the first time
        if images.shape[0]>=self.n_obs_steps:
            # add padding (from the start, since the most recent one is the last element i think)
            # need to add padding so that when folded we end up with the same number of elements
            # 0001234 for 4 so To-1
            images_pad=nn.functional.pad(images.swapaxes(0, 3),
                                         (self.n_obs_steps-1,0), "constant", 0).swapaxes(0,3)
            # images_pad=nn.functional.pad(images, (self.n_obs_steps-1,0), "constant", 0)
            images_shaped=images_pad.unfold(0, self.n_obs_steps, 1)
            images_shaped=images_shaped.permute(0,4,1,2,3)
            # images_shaped=images.reshape(-1, self.n_obs_steps, *images.shape[1:])
        else:
            images_shaped=images.reshape(-1,*images.shape)
        print(images_shaped.shape)
        # basically a weighted avg
        return (self.schedule.view(self.schedule.shape[0],1,1,1) * images_shaped).sum(dim=1)/self.schedule.sum() # way slower than the np version idk why
        # return np.average(images, axis=0, weights=self.schedule)
        
def test_create_afterimage():
    import skimage.io as io
    import numpy as np
    n=5
    imgs=[io.imread("brazil_renovated.jpg")]
    for i in range(1,n):
        imgs.append(np.roll(imgs[0],i*10,1))
    tt_one=torch.Tensor(np.array(imgs))
    tt=torch.concat([tt_one for i in range(2)])
    ag=AfterimageGenerator(n, "linear")
    res=ag.afterimage(tt)
    print(res.shape)
    return res
    

class MultiImageObsEncoderAfterimage(MultiImageObsEncoder):
    """Everything is the same as MultiImageObsEncoder, except to
    generate the encoding, we first compute an afterimage and pass
    that through the encoder. This means an additional n_obs_steps
    parameter is needed.
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
        super().__init__(shape_meta,
                         rgb_model,
                         resize_shape,
                         crop_shape,
                         random_crop,
                         use_group_norm,
                         share_rgb_model,
                         imagenet_norm)
        self.n_obs_steps = n_obs_steps
        self.afterimage_map=AfterimageGenerator(n_obs_steps, "linear")
    
    def forward(self, obs_dict):
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
                img = self.afterimage_map.afterimage(img)
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
                # print(f"imgshape {img.shape}") # yields torch.Size([1, 3, 240, 320])
                # or 128, 3, 240, 320
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.afterimage_map.afterimage(img)
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
                # print(f"lastimgshape {img.shape}") # yields torch.Size([1, 3, 216, 288])
                # print(f"featshape {feature.shape}") # yields torch.Size([1, 512]) or 128,512
        
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
