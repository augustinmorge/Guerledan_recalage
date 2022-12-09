#!/usr/bin/python3
import numpy as np
import os
file_path = os.path.dirname(os.path.abspath(__file__))

# def load_first_time_data():
print("Importing the log file..")
""" Import INS """
# """ Change the log """
# file = open("./resources/sbgCenterExport.txt", "r")
# file_text = file.readlines()
# newfile = open("./resources/sbgCenterExport_new.txt", "w")
# for i in range(250000,len(file_text)):
#     line = file_text[i]
#     newfile.write(line)
# file.close()
# newfile.close()
filepath = file_path+"/../resources/sbgCenterExport_new.txt"
data = np.loadtxt(filepath, dtype="U")
T = data[:,0]
LAT = np.float64(data[:,4])
LON = np.float64(data[:,5])
V_X = np.float64(data[:,1])
V_Y = np.float64(data[:,2])
V_Z = np.float64(data[:,3])

#Error on importation
LAT_STD = np.float64(data[:,6])
LON_STD = np.float64(data[:,7])
V_X_STD = np.float64(data[:,8])
V_Y_STD = np.float64(data[:,9])
V_Z_STD = np.float64(data[:,10])


# np.savez("ins.npz", T=T, LAT=LAT, LON=LON, V_X=V_X, V_Y=V_Y, V_Z=V_Z,\
#                     LAT_STD=LAT_STD, LON_STD=LON_STD, V_X_STD=V_X_STD,\
#                     V_Y_STD=V_Y_STD, V_Z_STD=V_Z_STD,\
#                     dtype = np.float64, precision = 16)

""" Import DVL """
## TODO:


""" Import the MNT """
AimeNT = np.loadtxt(file_path+"/../resources/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)
MNT = []
for i in AimeNT:
    MNT.append(i.split(','))
    MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])]
MNT = np.array(MNT)

# np.savez("mnt.npz", MNT=MNT, dtype = np.float64, precision = 16)

print("End of the importation.")



###

# load_first_time_data()
# """ Load the data from npy file """
#
# ins = np.load(file_path + "/ins.npz")
# T = ins['T']
# LAT = ins['LAT']
# LON = ins['LON']
# V_X = ins['V_X']
# V_Y = ins['V_Y']
# V_Z = ins['V_Y']
# LAT_STD = ins['LAT_STD']
# LON_STD = ins['LON_STD']
# V_X_STD = ins['V_X_STD']
# V_Y_STD = ins['V_Y_STD']
# V_Z_STD = ins['V_Z_STD']
#
# mnt = np.load(file_path + "/mnt.npz")
# MNT = mnt['MNT']


# if __name__ == '__main__':
#     import PIL.Image as Image
#     import osm_ui
#     import matplotlib.pyplot as plt
#     lac = Image.open("./imgs/ortho_sat_2016_guerledan.tif")
#     axes = osm_ui.plot_map(lac, (-3.118111, -2.954274), (48.183105, 48.237852), "Mis à l'eau de l'AUV")
#     osm_ui.plot_xy_add(axes, LON, LAT)
#     axes.legend(("ins",))
#     print("Start to display the log..")
#     plt.ion()
#     for i in range(0,LON.shape[0],10000):
#         if i == 0:
#             point = plt.scatter(LON[i,], LAT[i,], color = "red", label = "Panopée")
#         point = plt.scatter(LON[i,], LAT[i,], color = "red")
#         plt.pause(0.001)
#         plt.legend()
#         plt.show()
#         print(i)
