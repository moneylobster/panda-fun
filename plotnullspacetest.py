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
    print(data.shape)
    for i,val in enumerate(data):
        print(i,val)
        plt.subplot(7,1,i+1)
        plt.plot(val,label=name)

fig=plt.figure()

plot_joints(fig,jointposq, "jointpos")
plot_joints(fig,cartimpdefq, "cartimpdef")
plot_joints(fig,cartimpq, "cartimp")
plt.legend(loc="best")
fig.savefig("qvals.png")
