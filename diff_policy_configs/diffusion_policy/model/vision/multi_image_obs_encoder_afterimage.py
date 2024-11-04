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

def afterimage_schedule(sch_type, To):
    """Create a schedule. To is the obs. horizon.
    Schedule types:
    Linear
    """
    if sch_type=="linear":
        return np.linspace(0,1,To)
    else:
        raise NotImplementedError(f"Unsupported schedule type {sch_type}")

def create_afterimage(images, schedule):
    """Create an afterimage from images according to weight schedule.
    Schedule is a vector of weights."""
    return np.average(images, axis=0, weights=schedule)

def test_create_afterimage():
    import skimage.io as io
    import numpy as np
    imgs=[io.imread("brazil_renovated.jpg")]
    for i in range(1,10):
        imgs.append(np.roll(imgs[0],i*10,1))
    print(np.array(imgs).shape)
    res=create_afterimage(np.array(imgs), np.linspace(0,1,10))
    print(res.shape)
    return res
    

class MultiImageObsEncoderAfterimage(MultiImageObsEncoder):
    """Everything is the same as MultiImageObsEncoder, except to
    generate the encoding, we do not pass each image from the model,
    instead we compute an afterimage and pass that through.
    """
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
                print(f"imgshape {img.shape}")
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
                print(f"lastimgshape {img.shape}")
                print(f"featshape {feature.shape}")
        
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
