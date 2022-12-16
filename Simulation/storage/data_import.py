#!/usr/bin/python3
import numpy as np
import os
import pyproj
from sklearn.neighbors import KDTree
import joblib

file_path = os.path.dirname(os.path.abspath(__file__))

bool_txt = False
bool_compress = True

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
        # gcs = pyproj.Proj(init='epsg:4326') # Define the WGS84 GCS
        gcs = pyproj.CRS('epsg:4326')
        # proj = pyproj.Proj(init='epsg:2154') # Define the Lambert 93 projection
        proj = pyproj.CRS('epsg:2154')
        lon_mnt, lat_mnt = pyproj.transform(proj, gcs, MNT[:,0], MNT[:,1]) # Convert x and y values to latitude and longitude values
        MNT[:,0], MNT[:,1] = lon_mnt, lat_mnt


    wpt_ponton = (48.1989495, -3.0148023)
    def coord2cart(coords,coords_ref=wpt_ponton):
        R = 6372800
        ly,lx = coords
        lym,lxm = coords_ref
        x_tilde = R * np.cos(ly*np.pi/180)*(lx-lxm)*np.pi/180
        y_tilde = R * (ly-lym)*np.pi/180
        return np.array([x_tilde,y_tilde])

    print("Building KDTree..")
    nx_mnt, ny_mnt = coord2cart((MNT[:,1],MNT[:,0]))
    vec_mnt = np.vstack((nx_mnt, ny_mnt)).T
    kd_tree = KDTree(vec_mnt, metric="euclidean")

    """ Save the data """
    np.savez("ins.npz", T=T, LAT=LAT, LON=LON, V_X=V_X, V_Y=V_Y, V_Z=V_Z,\
    LAT_STD=LAT_STD, LON_STD=LON_STD, V_X_STD=V_X_STD,\
    V_Y_STD=V_Y_STD, V_Z_STD=V_Z_STD,\
    dtype = np.float64, precision = 16)
    np.savez("mnt.npz", MNT=MNT, dtype = np.float64, precision = 16)
    # with open('kd_tree.pkl', 'wb') as f:
    #     pickle.dump(kd_tree, f)
    with open('kd_tree.joblib', 'wb') as f:
        joblib.dump(kd_tree, f)

if bool_compress:
    """ Load the compressed data """

    """ Load npz file """
    print("Loading the compressed data..")
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

    """ Load the KD-Tree """
    # Load the KD tree object from the file
    # with open(file_path+'/kd_tree.pkl', 'rb') as f:
    #     kd_tree = pickle.load(f)
    with open(file_path+'/kd_tree.joblib', 'rb') as f:
        kd_tree = joblib.load(f)
