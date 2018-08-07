import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot2d(data, name, color):
    fig = plt.figure()
    plt.xlabel('Voxels')
    plt.ylabel('Voxels')
    plt.ylim(0, 512)
    plt.xlim(0, 512)
    x, y, z = data
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c = color, label = name, s = [0.25] * len(x))
    plt.savefig("%s.png" % name)

def plot3d(data, name, color):
    fig = plt.figure()
    plt.xlabel('Voxels')
    plt.ylabel('Voxels')
    plt.zlabel('Voxels')
    plt.xlim(0, 64)
    plt.ylim(0, 64)
    plt.zlim(0, 64)
    x, y, z = data
    ax = Axes3D(fig)
    ax.scatter(x, y, z, color = color, s = [1] * len(data[0]))
    plt.savefig("%s.png" % name)