import matplotlib.pyplot as plt
import numpy as np

jointposq=np.load("jointposq.npy")
cartimpdefq=np.load("cartimpdefq.npy")
cartimpq=np.load("cartimpq.npy")

### plotting

def plot_joints(fig,data,name):
    "plot all joints onto different subplots"
    data=np.array(data)
    data=data.reshape(7,-1)
    for i,val in enumerate(data.T):
        plt.subplot(7,1,i)
        plt.plot(val,name)

fig=plt.figure()

plot_joints(fig,jointposq, "jointpos")
plot_joints(fig,cartimpdefq, "cartimpdef")
plot_joints(fig,cartimpq, "cartimp")
fig.legend(loc="best")
fig.savefig("qvals.png")
