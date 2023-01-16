#!/usr/bin/python3
import numpy as np
import os
import pyproj
from sklearn.neighbors import KDTree
import joblib, pickle

file_path = os.path.dirname(os.path.abspath(__file__))

bool_txt = 1
bool_compress = 0
if bool_txt*bool_compress or not(bool_txt+bool_compress): import sys; print("Please choose txt or compressed data"); sys.exit()
data_cropped = 1

# Définit les coordonnées de référence
wpt_ponton = (48.1989495, -3.0148023)

def coord2cart(coords,coords_ref=wpt_ponton):
    R = 6372800
    ly,lx = coords
    lym,lxm = coords_ref
    x_tilde = R * np.cos(ly*np.pi/180)*(lx-lxm)*np.pi/180
    y_tilde = R * (ly-lym)*np.pi/180
    return np.array([x_tilde,y_tilde])

""" Uncomment the following section to crop the data from INS """
# file = open("/sbgCenterExport.txt", "r")
# file_text = file.readlines()
# newfile = open("/sbgCenterExport_new.txt", "w")
# for i in range(250000,len(file_text)):
#     line = file_text[i]
#     newfile.write(line)
# file.close()
# newfile.close()

if bool_txt:
    """ Import INS """
    print("Importing the INS-TXT file..")
    filepath = file_path+"/sbgCenterExport_new.txt"
    data = np.loadtxt(filepath, dtype="U")
    T = data[:,0]
    LAT = np.float64(data[:,4])
    LON = np.float64(data[:,5])
    V_X = np.float64(data[:,2])
    V_Y = np.float64(data[:,1])
    V_Z = np.float64(data[:,3])

    #Error on importation
    LAT_STD = np.float64(data[:,6])
    LON_STD = np.float64(data[:,7])
    V_X_STD = np.float64(data[:,9])
    V_Y_STD = np.float64(data[:,8])
    V_Z_STD = np.float64(data[:,10])


    """ Import DVL """
    print("Importing the DVL-TXT file..")
    ## TODO:


    print("Importing the MNT-TXT file..")
    """ Import the MNT """


    MNT = []
    if data_cropped: #Choose the txt file
        MNT_txt = np.loadtxt(file_path+"/../mnt/guerledan_cropped.txt", dtype = str)
        for i in MNT_txt:
            MNT.append(i.split(','))
            MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2])]
        MNT = np.array(MNT)

    else: #Choose the compressed file
        MNT_txt = np.loadtxt(file_path+"/../mnt/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)

        #Flip the MNT
        for i in MNT_txt:
            MNT.append(i.split(','))
            MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])]
        MNT = np.array(MNT)

        #Transform the proj
        gcs = pyproj.CRS('epsg:4326')
        proj = pyproj.CRS('epsg:2154')
        lat_mnt, lon_mnt = pyproj.transform(proj, gcs, MNT[:,0], MNT[:,1]) # Convert x and y values to latitude and longitude values
        MNT[:,0], MNT[:,1] = lon_mnt, lat_mnt


    print("Building KDTree..")
    nx_mnt, ny_mnt = coord2cart((MNT[:,1],MNT[:,0]))
    vec_mnt = np.vstack((nx_mnt, ny_mnt)).T
    kd_tree = KDTree(vec_mnt, metric="euclidean")

    """ Uncomment the following section to save the data """
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
