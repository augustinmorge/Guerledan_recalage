import numpy as np
import pyproj
import os
file_path = os.path.dirname(os.path.abspath(__file__))

with open("../resources/guerledan_cropped.txt","w") as mnt_cropped:
    AimeNT = np.loadtxt(file_path+"/../resources/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)
    MNT = []
    for i in AimeNT:
        MNT.append(i.split(','))
        MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])]
    MNT = np.array(MNT)

    gcs = pyproj.Proj(init='epsg:4326') # Define the WGS84 GCS
    proj = pyproj.Proj(init='epsg:2154') # Define the Lambert 93 projection
    lon_mnt, lat_mnt = pyproj.transform(proj, gcs, MNT[:,0], MNT[:,1]) # Convert x and y values to latitude and longitude values
    MNT[:,0], MNT[:,1] = lon_mnt, lat_mnt

    for i in range(MNT.shape[0]):
        x_mnt, y_mnt, z_mnt = MNT[i,0], MNT[i,1], MNT[i,2]
        if x_mnt > -3.02045 and 48.19654 < y_mnt < 48.20594:
            mnt_cropped.write(str(x_mnt)+"," + str(y_mnt) + "," + str(z_mnt)+"\n")
