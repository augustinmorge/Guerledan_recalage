#!/usr/bin/python3
import numpy as np
import pyproj
import os
file_path = os.path.dirname(os.path.abspath(__file__))

import warnings
warnings.filterwarnings("ignore")

with open("guerledan_cropped.txt","w") as mnt_cropped:
    MNT_txt = np.loadtxt(file_path+"/../mnt/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype=str)
    lines = np.array([line.split(",") for line in MNT_txt])
    cart_mnt_x, cart_mnt_y, cart_mnt_z0, cart_mnt_z1 = lines.T
    gcs = pyproj.CRS('epsg:4326')
    proj = pyproj.CRS('epsg:2154')
    lat_mnt, lon_mnt = pyproj.transform(proj, gcs, cart_mnt_x, cart_mnt_y)
    print(lon_mnt, lat_mnt)
    mask = (lon_mnt > -3.02045) & (48.19654 < lat_mnt) & (lat_mnt < 48.20594)
    cropped = lines[mask]
    np.savetxt(mnt_cropped, cropped, delimiter=",", fmt="%s")
