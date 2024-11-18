from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

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
    def __init__(self, afterimage_horizon, schedule_type):
        self.afterimage_horizon=afterimage_horizon
        self.schedule = self.create_schedule(afterimage_horizon, schedule_type)
        
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
        if images.shape[0]>=self.afterimage_horizon:
            # add padding (from the start, since the most recent one
            # is the last element i think) so that when folded we end
            # up with the same number of elements
            images_pad=nn.functional.pad(images.swapaxes(0, 3),
                                         (self.afterimage_horizon-1,0), "constant", 0).swapaxes(0,3)
            images_shaped=images_pad.unfold(0, self.afterimage_horizon, 1)
            images_shaped=images_shaped.permute(0,4,1,2,3)
        else:
            images_shaped=images.reshape(-1,*images.shape)
        # basically a weighted avg
        return (self.schedule.view(self.schedule.shape[0],1,1,1) * images_shaped).sum(dim=1)/self.schedule.sum() # way slower than the np version idk why
        
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
    
def plot_afterimage(imgs):
    import skimage.io as io
    plt.tight_layout()
    for i in range(imgs.shape[0]):
        plt.subplot(2,5,i+1)
        io.imshow(a[i].numpy().astype(np.int32))
