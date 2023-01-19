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
data_cropped = 0

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
# file = open("/../resources/sbgCenterExport.txt", "r")
# file_text = file.readlines()
# newfile = open("/../resources/sbgCenterExport_new.txt", "w")
# for i in range(250000,len(file_text)):
#     line = file_text[i]
#     newfile.write(line)
# file.close()
# newfile.close()

if bool_txt:
    """ Import INS """
    # print("Importing the INS-TXT file..")
    # filepath = file_path+"/sbgCenterExport_without_header.txt"
    # data = np.loadtxt(filepath, dtype="U")
    # T = data[:,0]
    # V_Y = np.float64(data[:,1])
    # V_X = np.float64(data[:,2])
    # V_Z = np.float64(data[:,3])
    # LAT = np.float64(data[:,4])
    # LON = np.float64(data[:,5])
    #
    # #Error on importation
    # LAT_STD = np.float64(data[:,6])
    # LON_STD = np.float64(data[:,7])
    # V_Y_STD = np.float64(data[:,8])
    # V_X_STD = np.float64(data[:,9])
    # V_Z_STD = np.float64(data[:,10])
    #
    # # #Attitude
    # # YAW = np.float64(data[:,11])
    # # YAW_STD = np.float64(data[:,12])
    # # ROLL = np.float64(data[:,13])
    # # ROLL_STD = np.float64(data[:,14])
    # # PITCH = np.float64(data[:,15])
    # # PITCH_STD = np.float64(data[:,16])


    """ Import DVL """
    print("Importing the DVL-TXT file..")
    filepath = file_path+"/dvl_afternoon"
    filepath_time = filepath+"/time.txt"
    filepath_bt_all = filepath+"/bt_all.txt"
    DVL = []
    data_dvl_time = np.genfromtxt(filepath_time, delimiter = ',')#np.loadtxt(filepath_time,dtype="U")
    data_dvl_bt_all = np.loadtxt(filepath_bt_all,dtype="U")
    DVL_T_Y = data_dvl_time[:,0]
    DVL_T_M = data_dvl_time[:,1]
    DVL_T_D = data_dvl_time[:,2]
    DVL_T_H = data_dvl_time[:,3]
    DVL_T_MIN = data_dvl_time[:,4]
    DVL_T_SEC = data_dvl_time[:,5]
    DVL_T_CSEC = data_dvl_time[:,6]
    DVL_T = 60*60*DVL_T_H + 60*DVL_T_MIN + DVL_T_SEC + 1/100.*DVL_T_CSEC
    print(DVL_T)

    # DVL_V_X =
    # DVL_V_Y =
    # DVL_V_Z =

    # DVL = np.column_stack(DVL_T, DVL_V_X, DVL_V_Y, DVL_V_Z)

    """ Import MBES """
    print("Importing the MBES-TXT file..")
    filepath = file_path+"/mbes_navsight.txt"
    # Lire les données à partir d'un fichier
    data_mbes = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype = "U")

    # Séparer les colonnes en utilisant les index
    Beam = np.float64(data_mbes[:, 0])
    Date_Time = data_mbes[:, 1]
    Footprint_X = np.float64(data_mbes[:, 2])
    Footprint_Y = np.float64(data_mbes[:, 3])
    Footprint_Z = np.float64(data_mbes[:, 4])
    print(Beam)
    print(Date_Time)
    print(Footprint_X)
    print(Footprint_Y)
    print(Footprint_Z)


    print("Importing the MNT-TXT file..")
    """ Import the MNT """

    # MNT = []
    # if data_cropped: #Choose the txt file
    #     MNT_txt = np.loadtxt(file_path+"/../mnt/guerledan_cropped.txt", dtype = str)
    #     for i in MNT_txt:
    #         MNT.append(i.split(','))
    #         MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2])]
    #     MNT = np.array(MNT)
    #
    # else: #Choose the compressed file
    #     MNT_txt = np.loadtxt(file_path+"/../mnt/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)
    #
    #     #Flip the MNT
    #     for i in MNT_txt:
    #         MNT.append(i.split(','))
    #         MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])]
    #     MNT = np.array(MNT)
    #
    #     #Transform the proj
    #     gcs = pyproj.CRS('epsg:4326')
    #     proj = pyproj.CRS('epsg:2154')
    #     lat_mnt, lon_mnt = pyproj.transform(proj, gcs, MNT[:,0], MNT[:,1]) # Convert x and y values to latitude and longitude values
    #     MNT[:,0], MNT[:,1] = lon_mnt, lat_mnt
    #
    #
    # print("Building KDTree..")
    # nx_mnt, ny_mnt = coord2cart((MNT[:,1],MNT[:,0]))
    # vec_mnt = np.vstack((nx_mnt, ny_mnt)).T
    # kd_tree = KDTree(vec_mnt, metric="euclidean")
    #
    # """ Uncomment the following section to save the data """
    # np.savez("ins.npz", T=T, LAT=LAT, LON=LON, V_X=V_X, V_Y=V_Y, V_Z=V_Z,\
    #         LAT_STD=LAT_STD, LON_STD=LON_STD, V_X_STD=V_X_STD,\
    #         V_Y_STD=V_Y_STD, V_Z_STD=V_Z_STD, YAW=YAW, YAW_STD=YAW_STD,\
    #         ROLL=ROLL, ROLL_STD=ROLL_STD, PITCH=PITCH, PITCH_STD=PITCH_STD,\
    #         dtype = np.float64, precision = 16)
    # np.savez("mnt.npz", MNT=MNT, dtype = np.float64, precision = 16)
    # # with open('kd_tree.pkl', 'wb') as f:
    # #     pickle.dump(kd_tree, f)
    # with open('kd_tree.joblib', 'wb') as f:
    #     joblib.dump(kd_tree, f)

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

    # YAW = ins['YAW']
    # YAW_STD = ins['YAW_STD']
    # ROLL = ins['ROLL']
    # ROLL_STD = ins['ROLL_STD']
    # PITCH = ins['PITCH']
    # PITCH_STD = ins['PITCH_STD']

    mnt = np.load(file_path + "/mnt.npz")
    MNT = mnt['MNT']

    """ Load the KD-Tree """
    # Load the KD tree object from the file
    # with open(file_path+'/kd_tree.pkl', 'rb') as f:
    #     kd_tree = pickle.load(f)
    with open(file_path+'/kd_tree.joblib', 'rb') as f:
        kd_tree = joblib.load(f)
