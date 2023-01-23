#!/usr/bin/python3
import numpy as np
import os
import pyproj
from sklearn.neighbors import KDTree
import joblib, pickle

file_path = os.path.dirname(os.path.abspath(__file__))

bool_txt = 0
bool_compress = 1
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
    print("Importing the INS-TXT file..")
    filepath = file_path+"/sbgCenterExport_without_header.txt"
    data = np.loadtxt(filepath, dtype="U")
    T = data[:,0]
    T = np.array([dt.split(":") for dt in T], dtype=np.float64)
    T = 60*60*T[:,0] + 60*T[:,1] + T[:,2]
    V_Y = np.float64(data[:,1])
    V_X = np.float64(data[:,2])
    V_Z = np.float64(data[:,3])
    LAT = np.float64(data[:,4])
    LON = np.float64(data[:,5])

    #Error on importation
    LAT_STD = np.float64(data[:,6])
    LON_STD = np.float64(data[:,7])
    V_Y_STD = np.float64(data[:,8])
    V_X_STD = np.float64(data[:,9])
    V_Z_STD = np.float64(data[:,10])

    # #Attitude
    YAW = np.float64(data[:,11])
    YAW_STD = np.float64(data[:,12])
    ROLL = np.float64(data[:,13])
    ROLL_STD = np.float64(data[:,14])
    PITCH = np.float64(data[:,15])
    PITCH_STD = np.float64(data[:,16])


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
    print("DVL_T: ",DVL_T)

    # DVL_V_X =
    # DVL_V_Y =
    # DVL_V_Z =

    # DVL = np.column_stack(DVL_T, DVL_V_X, DVL_V_Y, DVL_V_Z)

    """ Import MBES """
    print("Importing the MBES-TXT file..")
    filepath = file_path+"/mbes_navsight.txt"
    # Read data from file
    data_mbes = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype = "U")

    # Split columns using index
    Beam, Date_Time, Footprint_X, Footprint_Y, Footprint_Z = data_mbes[:, 0], data_mbes[:, 1], data_mbes[:, 2], data_mbes[:, 3], data_mbes[:, 4]

    # Convert columns to float
    Beam, Footprint_X, Footprint_Y, Footprint_Z = np.float64(Beam), np.float64(Footprint_X), np.float64(Footprint_Y), np.float64(Footprint_Z)

    # Create Time_MBES array
    Time_MBES = np.array([dt.split(" ")[1].split(":") for dt in Date_Time], dtype=np.float64)

    # Create the TIME vector array
    Time_MBES_seconds = (60*60*Time_MBES[:,0] + 60*Time_MBES[:,1] + Time_MBES[:,2])
    print("Beam: ",Beam)
    print("Footprint_X: ",Footprint_X)
    print("Footprint_Y: ",Footprint_Y)
    print("Footprint_Z: ",Footprint_Z)
    print("Time_MBES_seconds: ",Time_MBES_seconds)


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
            V_Y_STD=V_Y_STD, V_Z_STD=V_Z_STD, YAW=YAW, YAW_STD=YAW_STD,\
            ROLL=ROLL, ROLL_STD=ROLL_STD, PITCH=PITCH, PITCH_STD=PITCH_STD,\
            dtype = np.float64, precision = 16)
    np.savez("mnt.npz", MNT=MNT, dtype = np.float64, precision = 16)
    # with open('kd_tree.pkl', 'wb') as f:
    #     pickle.dump(kd_tree, f)
    with open('kd_tree.joblib', 'wb') as f:
        joblib.dump(kd_tree, f)
    np.savez('mbes.npz', BEAMS = Beam,
                        MBES_X = Footprint_X,
                        MBES_Y = Footprint_Y,
                        MBES_Z = Footprint_Z,
                        MBES_T = Time_MBES_seconds, dtype = np.float64)

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


    """ Load the MBES """
    # Load data from npz file
    mbes = np.load(file_path + "/mbes.npz")
    BEAMS, MBES_X, MBES_Y, MBES_Z, MBES_T = mbes['BEAMS'], mbes['MBES_X'], mbes['MBES_Y'], mbes['MBES_Z'], mbes['MBES_T']

    # Find the indices where the value of BEAMS decreases
    indices = np.where(np.diff(BEAMS) < 0)[0]

    # Append the last index to indices
    indices = np.append(indices, BEAMS.shape[0]-1)

    # Use the indices to extract the corresponding values of other arrays
    mid_indices = (indices[:-1]+indices[1:])//2

    MBES_T = np.array(MBES_T[mid_indices])
    MBES_X = np.array(MBES_X[mid_indices])
    MBES_Y = np.array(MBES_Y[mid_indices])
    MBES_Z = np.array(MBES_Z[mid_indices])

    """ Interpolate data """
    from scipy.interpolate import interp1d

    # Déterminez les temps de début et de fin communs entre T et MBES_T
    start_time = max(T[0], MBES_T[0])
    end_time = min(T[-1], MBES_T[-1])

    # # Créez un nouveau vecteur de temps avec un temps de début commun, un temps de fin commun et un pas de temps fixe
    # # dt = 0.1 #max(MBES_T[1,]-MBES[0,], T[1,] - T[0,]) # pas de temps en secondes
    # # Calculer la différence de temps entre chaque mesure
    # time_diffs_mbes = np.diff(MBES_T)
    # # Calculer la fréquence moyenne en divisant la différence de temps par le nombre de mesures
    # mean_freq_mbes = 1 / (np.mean(time_diffs_mbes))
    # freq_ins = 1/0.005
    # f_low = min(freq_ins, mean_freq_mbes)
    # dt = 1/f_low
    # print(dt)

    dt = 0.1

    T_glob = np.arange(start_time, end_time, dt)

    # Interpolez les données de T sur le nouveau vecteur de temps T_glob
    f_T = interp1d(T, T)
    T_interp = f_T(T_glob)

    # Interpolez les autres variables (x, y, z, vitesse, etc.) de la même manière
    f_LAT = interp1d(T, LAT)
    f_LON = interp1d(T, LON)
    f_V_X = interp1d(T, V_X)
    f_V_Y = interp1d(T, V_Y)
    f_V_Z = interp1d(T, V_Z)
    f_LAT_STD = interp1d(T, LAT_STD)
    f_LON_STD = interp1d(T, LON_STD)
    f_V_X_STD = interp1d(T, V_X_STD)
    f_V_Y_STD = interp1d(T, V_Y_STD)
    f_V_Z_STD = interp1d(T, V_Z_STD)

    LAT_interp = f_LAT(T_glob)
    LON_interp = f_LON(T_glob)
    V_X_interp = f_V_X(T_glob)
    V_Y_interp = f_V_Y(T_glob)
    V_Z_interp = f_V_Z(T_glob)
    LAT_STD_interp = f_LAT_STD(T_glob)
    LON_STD_interp = f_LON_STD(T_glob)
    V_X_STD_interp = f_V_X_STD(T_glob)
    V_Y_STD_interp = f_V_Y_STD(T_glob)
    V_Z_STD_interp = f_V_Z_STD(T_glob)

    # Interpolate the MBES
    f_MBES_T = interp1d(MBES_T, MBES_T)
    f_MBES_X = interp1d(MBES_T, MBES_X)
    f_MBES_Y = interp1d(MBES_T, MBES_Y)
    f_MBES_Z = interp1d(MBES_T, MBES_Z)

    MBES_T_interp = f_MBES_T(T_glob)
    MBES_X_interp = f_MBES_X(T_glob)
    MBES_Y_interp = f_MBES_Y(T_glob)
    MBES_Z_interp = f_MBES_Z(T_glob)

    # Update the previous vector
    T = T_interp
    MBES_T = MBES_T_interp
    LAT = LAT_interp
    LON = LON_interp
    V_X = V_X_interp
    V_Y = V_Y_interp
    V_Z = V_Z_interp
    LAT_STD = LAT_STD_interp
    LON_STD = LON_STD_interp
    V_X_STD = V_X_STD_interp
    V_Y_STD = V_Y_STD_interp
    V_Z_STD = V_Z_STD_interp

    MBES_T = MBES_T_interp
    MBES_X = MBES_X_interp
    MBES_Y = MBES_Y_interp
    MBES_Z = MBES_Z_interp
