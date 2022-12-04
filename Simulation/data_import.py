#!/usr/bin/python3
import numpy as np

""" Import INS """
filepath = "./resources/sbgCenterExport.txt"
data = np.loadtxt(filepath, dtype="U")
T = data[:,0]
V_X = np.float64(data[:,1])
V_Y = np.float64(data[:,2])
V_Z = np.float64(data[:,3])
LAT = np.float64(data[:,4])
LON = np.float64(data[:,5])
LAT_STD = np.float64(data[:,6])
LON_STD = np.float64(data[:,7])
V_X_STD = np.float64(data[:,8])
V_Y_STD = np.float64(data[:,9])
V_Z_STD = np.float64(data[:,10])

""" Import DVL """
## TODO:

if __name__ == '__main__':
    import PIL.Image as Image
    import osm_ui
    import matplotlib.pyplot as plt
    lac = Image.open("./imgs/ortho_sat_2016_guerledan.tif")
    axes = osm_ui.plot_map(lac, (-3.118111, -2.954274), (48.183105, 48.237852), "Mis à l'eau de l'AUV")
    osm_ui.plot_xy_add(axes, LON, LAT)
    axes.legend(("ins",))
    plt.show()
