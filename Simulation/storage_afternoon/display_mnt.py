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
    cb = fig.colorbar(sc)

    # Set the x and y axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.plot(LON,LAT,color='red',label='trajectory')

    # Show the plot
    plt.savefig("MNT_G1.png", dpi = 200) #augmenter dpi pour une meilleure résolution
    # plt.show()
    plt.legend()




if __name__ == '__main__':
    from data_import import *
    # print(MNT[:,0])
    # print(np.min(LON),np.max(LON))
    # print(np.min(LAT),np.max(LAT))
    LON, LAT = coord2cart((LAT,LON))
    display_mnt(LON, LAT, MNT)

    from PIL import Image
    image = Image.open("MNT_G1.png")
    image.show()


    # import PIL.Image as Image
    # import osm_ui
    # import matplotlib.pyplot as plt
    # lac = Image.open("../imgs/ortho_sat_2016_guerledan.tif")
    # axes = osm_ui.plot_map(lac, (-3.118111, -2.954274), (48.183105, 48.237852), "Mis à l'eau de l'AUV")
    # osm_ui.plot_xy_add(axes, LON, LAT)
    # axes.legend(("ins",))
    # print("Start to display the log..")
    # plt.ion()
    # for i in range(0,LON.shape[0],10000):
    #     if i == 0:
    #         point = plt.scatter(LON[i,], LAT[i,], color = "red", label = "Panopée")
    #     point = plt.scatter(LON[i,], LAT[i,], color = "red")
    #     plt.pause(0.001)
    #     plt.legend()
    #     plt.show()
    #     print(i)
