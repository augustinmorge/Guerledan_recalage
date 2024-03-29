#!/usr/bin/python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def display_mnt(LON, LAT, mnt):
    # Extract x and y values from the mnt array
    x = mnt[:,0]
    y = mnt[:,1]

    # Create a new figure
    fig, ax = plt.subplots()

    # Extract z values from the mnt array
    z = -(mnt[:,2] - np.min(mnt[:,2]))

    # Create a scatter plot of the x and y values, colored by the z values
    sc = ax.scatter(x, y, c=z, cmap="terrain", s = 0.01)

    # Add a colorbar to the plot
    cb = fig.colorbar(sc) #, orientation = 'horizontal')

    # Set the x and y axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # ax.set_xlim(np.min(LON), np.max(LON))
    # ax.set_ylim(np.min(LAT), np.max(LAT))

    ax.set_aspect('equal')

    # plt.plot(LON,LAT,color='red',label='trajectory')

    # Show the plot
    plt.savefig("../mnt/MNT_2019.png", dpi = 1000, transparent = True) #augmenter dpi pour une meilleure résolution
    # plt.show()
    plt.legend()



if __name__ == '__main__':
    from data_import import *
    LON, LAT = coord2cart((LAT,LON))
    display_mnt(LON, LAT, MNT)

    from PIL import Image
    image = Image.open("MNT_G1.png")
    image.show()
