#!/usr/bin/python3
import numpy as np
import os
import pyproj

file_path = os.path.dirname(os.path.abspath(__file__))

bool_txt = True
bool_npz = False

if bool_txt:
    print("Importing the DVL-TXT file..")
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


    """ Import DVL """
    ## TODO:


    print("Importing the MNT-TXT file..")
    """ Import the MNT """
    data_cropped = True #((str(input("cropped ?[Y/] "))) == "Y")
    AimeNT = np.loadtxt(file_path+"/../resources/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)
    if data_cropped:
        AimeNT = np.loadtxt(file_path+"/../resources/guerledan_cropped.txt", dtype = str)
    MNT = []
    for i in AimeNT:
        MNT.append(i.split(','))
        if data_cropped:
            MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2])]
        else:
            MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])]
    MNT = np.array(MNT)

    if not data_cropped:
        gcs = pyproj.Proj(init='epsg:4326') # Define the WGS84 GCS
        proj = pyproj.Proj(init='epsg:2154') # Define the Lambert 93 projection
        lon_mnt, lat_mnt = pyproj.transform(proj, gcs, MNT[:,0], MNT[:,1]) # Convert x and y values to latitude and longitude values
        MNT[:,0], MNT[:,1] = lon_mnt, lat_mnt

    """ Save the data """
    # np.savez("ins.npz", T=T, LAT=LAT, LON=LON, V_X=V_X, V_Y=V_Y, V_Z=V_Z,\
    # LAT_STD=LAT_STD, LON_STD=LON_STD, V_X_STD=V_X_STD,\
    # V_Y_STD=V_Y_STD, V_Z_STD=V_Z_STD,\
    # dtype = np.float64, precision = 16)
    # np.savez("mnt.npz", MNT=MNT, dtype = np.float64, precision = 16)

    print("End of the importation.")

if bool_npz:
    """ Load the data from npy file """

    ins = np.load(file_path + "/ins.npz")
    T = ins['T']
    LAT = ins['LAT']
    LON = ins['LON']
    V_X = ins['V_X']
    V_Y = ins['V_Y']
    V_Z = ins['V_Y']
    LAT_STD = ins['LAT_STD']
    LON_STD = ins['LON_STD']
    V_X_STD = ins['V_X_STD']
    V_Y_STD = ins['V_Y_STD']
    V_Z_STD = ins['V_Z_STD']

    mnt = np.load(file_path + "/mnt.npz")
    MNT = mnt['MNT']

# if __name__ == '__main__':
#     import pyproj
#     x_mnt = MNT[:,0]
#     y_mnt = MNT[:,1]
#     gcs = pyproj.Proj(init='epsg:4326') # Define the WGS84 GCS
#     proj = pyproj.Proj(init='epsg:2154') # Define the Lambert 93 projection
#     lon_mnt, lat_mnt = pyproj.transform(proj, gcs, x_mnt, y_mnt) # Convert x and y values to latitude and longitude values
#     with open("data_mnt_2013.csv","w") as data:
#         for i in range(lon_mnt.shape[0]):
#             data.write(str(lon_mnt[i,])+";"+str(lat_mnt[i,])+"\n")
