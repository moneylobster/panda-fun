from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.afterimage_utils import AfterimageGenerator
import IPython
import plotly
import plotly.express as px

def show(img):
    fig=px.imshow(img.cpu().transpose(0,2))
    plotly.offline.plot(fig,filename="plot.html")

class MultiImageObsEncoderAfterimage(MultiImageObsEncoder):
    """Everything is the same as MultiImageObsEncoder, except to
    generate the encoding, we first compute an afterimage and pass
    that through the encoder. This means an additional
    afterimage_horizon parameter is needed.
    """

    def __init__(self,
                 shape_meta: dict,
                 rgb_model: Union[nn.Module, Dict[str,nn.Module]],
                 n_obs_steps,
                 afterimage_horizon,
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
        self.afterimage_horizon = afterimage_horizon
        self.afterimage_map = AfterimageGenerator(afterimage_horizon, n_obs_steps, "linear")
        print(f" HELLO IM the encoder and i have aft as {afterimage_horizon} and To as {n_obs_steps}")
    
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
                print(f"imgshape {img.shape}")
                # should yield torch.Size([aft_horizon+To-1(*batch if training), 3, 240, 320])
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.afterimage_map.afterimage(img)
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
                IPython.embed()
        
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
