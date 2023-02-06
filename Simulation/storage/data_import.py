#!/usr/bin/python3
import numpy as np
import os
import pyproj
from sklearn.neighbors import KDTree
import joblib, pickle

file_path = os.path.dirname(os.path.abspath(__file__))

bool_txt = 0
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
        # UTC Time	North Velocity	East Velocity	Down Velocity	Latitude	Longitude
        	# Latitude Std.	Longitude Std.	North Velocity Std.	East Velocity Std.
                # Down Velocity Std.	Yaw	Yaw Std.	Roll	Roll Std.	Pitch
                    # Pitch Std.

    filepath = file_path+"/IMU_trdi.txt"
    data_ins = np.genfromtxt(filepath, delimiter='\t', skip_header=2, dtype = "U")
    T = data_ins[:,0]
    T = np.array([dt.split(":") for dt in T], dtype=np.float64)
    T = 60*60*T[:,0] + 60*T[:,1] + T[:,2]

    V_Y = np.float64(data_ins[:,1])
    V_X = np.float64(data_ins[:,2])
    V_Z = np.float64(data_ins[:,3])
    LAT = np.float64(data_ins[:,4])
    LON = np.float64(data_ins[:,5])

    #Error on importation
    LAT_STD = np.float64(data_ins[:,6])
    LON_STD = np.float64(data_ins[:,7])
    V_Y_STD = np.float64(data_ins[:,8])
    V_X_STD = np.float64(data_ins[:,9])
    V_Z_STD = np.float64(data_ins[:,10])

    # #Attitude
    YAW = np.float64(data_ins[:,11])
    YAW_STD = np.float64(data_ins[:,12])
    ROLL = np.float64(data_ins[:,13])
    ROLL_STD = np.float64(data_ins[:,14])
    PITCH = np.float64(data_ins[:,15])
    PITCH_STD = np.float64(data_ins[:,16])

    ACC_X = np.float64(data_ins[:,17])
    ACC_Y = np.float64(data_ins[:,18])
    ACC_Z = np.float64(data_ins[:,19])
    GYR_X = np.float64(data_ins[:,20])
    GYR_Y = np.float64(data_ins[:,21])
    GYR_Z = np.float64(data_ins[:,22])

    """ Import DVL """
    print("Importing the DVL-TXT file..")
    filepath = file_path + "/dvl_trdi.txt"
    data_dvl = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    dvl_ensemble = data_dvl[:,0]
    dvl_Y = data_dvl[:,0]
    dvl_M = data_dvl[:,1]
    dvl_D = data_dvl[:,2]
    dvl_H = data_dvl[:,3]
    dvl_MN = data_dvl[:,4]
    dvl_S = data_dvl[:,5]
    dvl_CS = data_dvl[:,6]
    dvl_T = 60*60*dvl_H + 60*dvl_MN + dvl_S + 1/100.*dvl_CS

    dvl_BM1R = data_dvl[:,7]
    dvl_BM2R = data_dvl[:,8]
    dvl_BM3R = data_dvl[:,9]
    dvl_BM4R = data_dvl[:,10]
    dvl_VE = data_dvl[:,11]
    dvl_VN = data_dvl[:,12]
    dvl_VZ = data_dvl[:,13]
    dvl_VSTD = data_dvl[:,14]

    """ Import MBES """
    print("Importing the MBES-TXT file..")
    filepath = file_path+"/mbes_trdi.txt"
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
    # print("Beam: ",Beam)
    # print("Footprint_X: ",Footprint_X)
    # print("Footprint_Y: ",Footprint_Y)
    # print("Footprint_Z: ",Footprint_Z)
    # print("Time_MBES_seconds: ",Time_MBES_seconds)


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
    print("Saving the data into compressed ones")
    np.savez("ins.npz", T=T, LAT=LAT, LON=LON, V_X=V_X, V_Y=V_Y, V_Z=V_Z,\
            LAT_STD=LAT_STD, LON_STD=LON_STD, V_X_STD=V_X_STD,\
            V_Y_STD=V_Y_STD, V_Z_STD=V_Z_STD, YAW=YAW, YAW_STD=YAW_STD,\
            ROLL=ROLL, ROLL_STD=ROLL_STD, PITCH=PITCH, PITCH_STD=PITCH_STD,\
            ACC_X = ACC_X, ACC_Y = ACC_Y, ACC_Z = ACC_Z, GYR_X = GYR_X, GYR_Y = GYR_Y, GYR_Z = GYR_Z,\
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

    np.savez('dvl.npz', dvl_T = dvl_T,
                        dvl_BM1R = dvl_BM1R,
                        dvl_BM2R = dvl_BM2R,
                        dvl_BM3R = dvl_BM3R,
                        dvl_BM4R = dvl_BM4R,
                        dvl_VE = dvl_VE,
                        dvl_VN = dvl_VN,
                        dvl_VZ = dvl_VZ,
                        dvl_VSTD = dvl_VSTD, dtype = np.float64)

    print("Compressed data saved !")

""" Load the compressed data """

""" Load INS """
print("Loading the compressed data..")
ins = np.load(file_path + "/ins.npz")
T = ins['T']
LAT = ins['LAT']
LON = ins['LON']
V_X = ins['V_X']
V_Y = ins['V_Y']
V_Z = ins['V_Z']
LAT_STD = ins['LAT_STD']
LON_STD = ins['LON_STD']
V_X_STD = ins['V_X_STD']
V_Y_STD = ins['V_Y_STD']
V_Z_STD = ins['V_Z_STD']

YAW = ins['YAW']/180*np.pi
YAW_STD = ins['YAW_STD']/180*np.pi
ROLL = ins['ROLL']/180*np.pi
ROLL_STD = ins['ROLL_STD']/180*np.pi
PITCH = ins['PITCH']/180*np.pi
PITCH_STD = ins['PITCH_STD']/180*np.pi

ACC_X = ins['ACC_X']
ACC_Y = ins['ACC_Y']
ACC_Z = ins['ACC_Z']
GYR_X = ins['GYR_X']
GYR_Y = ins['GYR_Y']
GYR_Z = ins['GYR_Z']

""" Load INS """
mnt = np.load(file_path + "/mnt.npz")
MNT = mnt['MNT']
if bool_txt:
    nx_mnt, ny_mnt = coord2cart((MNT[:,1],MNT[:,0]))
    MNT[:,0] = nx_mnt
    MNT[:,1] = ny_mnt


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

# Use only the middle of the beams
MBES_T = np.array(MBES_T[(indices[:-1]+indices[1:])//2])
MBES_X = np.array(MBES_X[(indices[:-1]+indices[1:])//2])
MBES_Y = np.array(MBES_Y[(indices[:-1]+indices[1:])//2])
MBES_Z = np.array(MBES_Z[(indices[:-1]+indices[1:])//2])

# Use the average on all the beams
# MBES_T = np.array([np.mean(MBES_T[indices[i]:indices[i+1]]) for i in range(len(indices)-1)])
# MBES_X = np.array([np.mean(MBES_X[indices[i]:indices[i+1]]) for i in range(len(indices)-1)])
# MBES_Y = np.array([np.mean(MBES_Y[indices[i]:indices[i+1]]) for i in range(len(indices)-1)])
# MBES_Z = np.array([np.mean(MBES_Z[indices[i]:indices[i+1]]) for i in range(len(indices)-1)])

""" Load the DVL """
dvl = np.load(file_path + "/dvl.npz")
dvl_T = dvl["dvl_T"]
dvl_BM1R = dvl["dvl_BM1R"]/100. #en cm -> m
dvl_BM2R = dvl["dvl_BM2R"]/100. #en cm -> m
dvl_BM3R = dvl["dvl_BM3R"]/100. #en cm -> m
dvl_BM4R = dvl["dvl_BM4R"]/100. #en cm -> m
# dvl_VE = dvl["dvl_VE"]/1100.
# dvl_VN = dvl["dvl_VN"]/11000.
# dvl_VZ = dvl["dvl_VZ"]/11000.
# dvl_VSTD = dvl["dvl_VSTD"]/11000.
dvl_VE = dvl["dvl_VE"]/1000.
dvl_VN = dvl["dvl_VN"]/10000.
dvl_VZ = dvl["dvl_VZ"]/10000.
dvl_VSTD = dvl["dvl_VSTD"]/10000.

""" Interpolate data """
from scipy.interpolate import interp1d
apply_modif = True
if apply_modif:
    #Apply a mask
    mask_dvl_VE = np.abs(dvl_VE - np.mean(dvl_VE)) > 3*np.std(dvl_VE)
    mask_dvl_VN = np.abs(dvl_VN - np.mean(dvl_VN)) > 3*np.std(dvl_VN)
    mask = (mask_dvl_VE) | (mask_dvl_VN) | (dvl_BM1R == 0) | (dvl_BM2R == 0) | (dvl_BM3R == 0) | (dvl_BM4R == 0)

    print("Total point of DVL taken = ",dvl_VE[~mask].shape[0]/dvl_VE.shape[0]*100,"%")
    dvl_VE = dvl_VE[~mask]
    dvl_VN = dvl_VN[~mask]
    dvl_T = dvl_T[~mask]
    dvl_VZ = dvl_VZ[~mask]
    dvl_VSTD = dvl_VSTD[~mask]
    dvl_BM1R = dvl_BM1R[~mask]
    dvl_BM2R = dvl_BM2R[~mask]
    dvl_BM3R = dvl_BM3R[~mask]
    dvl_BM4R = dvl_BM4R[~mask]

# Déterminez les temps de début et de fin communs entre T et MBES_T
start_time = max(max(T[0], MBES_T[0]),dvl_T[0])
end_time = min(min(T[-1], MBES_T[-1]),dvl_T[-1])
global dt_br
# dt_br = 0.1 #np.mean(np.diff(dvl_T))
dt_br = np.mean(np.diff(dvl_T))
# dt = 0.2 #np.mean(np.diff(MBES_T))
# dt = 0.05 #ins

print(f"dt choosen = {dt_br}")
T_glob = np.arange(start_time, end_time, dt_br)

# Interpolez les données de T sur le nouveau vecteur de temps T_glob

# Interpolate the INS
f_T = interp1d(T, T)
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
f_YAW = interp1d(T,YAW)
f_YAW_STD = interp1d(T,YAW_STD)
f_ROLL = interp1d(T,ROLL)
f_ROLL_STD = interp1d(T,ROLL_STD)
f_PITCH = interp1d(T,PITCH)
f_PITCH_STD = interp1d(T,PITCH_STD)
f_ACC_X = interp1d(T,ACC_X)
f_ACC_Y = interp1d(T,ACC_Y)
f_ACC_Z = interp1d(T,ACC_Z)
f_GYR_X = interp1d(T,GYR_X)
f_GYR_Y = interp1d(T,GYR_Y)
f_GYR_Z = interp1d(T,GYR_Z)

T_interp = f_T(T_glob)
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
YAW_interp = f_YAW(T_glob)
YAW_STD_interp = f_YAW_STD(T_glob)
ROLL_interp = f_ROLL(T_glob)
ROLL_STD_interp = f_ROLL_STD(T_glob)
PITCH_interp = f_PITCH(T_glob)
PITCH_STD_interp = f_PITCH_STD(T_glob)
ACC_X_interp = f_ACC_X(T_glob)
ACC_Y_interp = f_ACC_Y(T_glob)
ACC_Z_interp = f_ACC_Z(T_glob)
GYR_X_interp = f_GYR_X(T_glob)
GYR_Y_interp = f_GYR_Y(T_glob)
GYR_Z_interp = f_GYR_Z(T_glob)

# Interpolate the MBES
f_MBES_T = interp1d(MBES_T, MBES_T)
f_MBES_X = interp1d(MBES_T, MBES_X)
f_MBES_Y = interp1d(MBES_T, MBES_Y)
f_MBES_Z = interp1d(MBES_T, MBES_Z)

MBES_T_interp = f_MBES_T(T_glob)
MBES_X_interp = f_MBES_X(T_glob)
MBES_Y_interp = f_MBES_Y(T_glob)
MBES_Z_interp = f_MBES_Z(T_glob)

# Interpolate the DVL
f_dvl_T = interp1d(dvl_T,dvl_T)
f_dvl_BM1R = interp1d(dvl_T,dvl_BM1R)
f_dvl_BM2R = interp1d(dvl_T,dvl_BM2R)
f_dvl_BM3R = interp1d(dvl_T,dvl_BM3R)
f_dvl_BM4R = interp1d(dvl_T,dvl_BM4R)
f_dvl_VE = interp1d(dvl_T,dvl_VE)
f_dvl_VN = interp1d(dvl_T,dvl_VN)
f_dvl_VZ = interp1d(dvl_T,dvl_VZ)
f_dvl_VSTD = interp1d(dvl_T,dvl_VSTD)

dvl_T_interp = f_dvl_T(T_glob)
dvl_BM1R_interp = f_dvl_BM1R(T_glob)
dvl_BM2R_interp = f_dvl_BM2R(T_glob)
dvl_BM3R_interp = f_dvl_BM3R(T_glob)
dvl_BM4R_interp = f_dvl_BM4R(T_glob)
dvl_VE_interp = f_dvl_VE(T_glob)
dvl_VN_interp = f_dvl_VN(T_glob)
dvl_VZ_interp = f_dvl_VZ(T_glob)
dvl_VSTD_interp = f_dvl_VSTD(T_glob)

# Update the previous vector
#INS
T = T_interp
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
YAW = YAW_interp
YAW_STD = YAW_STD_interp
ROLL = ROLL_interp
ROLL_STD = ROLL_STD_interp
PITCH = PITCH_interp
PITCH_STD = PITCH_STD_interp
ACC_X = ACC_X_interp
ACC_Y = ACC_Y_interp
ACC_Z = ACC_Z_interp
GYR_X = GYR_X_interp
GYR_Y = GYR_Y_interp
GYR_Z = GYR_Z_interp

#MBES
MBES_T = MBES_T_interp
MBES_X = MBES_X_interp
MBES_Y = MBES_Y_interp
MBES_Z = MBES_Z_interp

#DVL
dvl_T = dvl_T_interp
dvl_BM1R = dvl_BM1R_interp
dvl_BM2R = dvl_BM2R_interp
dvl_BM3R = dvl_BM3R_interp
dvl_BM4R = dvl_BM4R_interp
dvl_VE = dvl_VE_interp
dvl_VN = dvl_VN_interp
dvl_VZ = dvl_VZ_interp
dvl_VSTD = dvl_VSTD_interp

YAW = -YAW+np.pi/2
#Rotate the dvl
angle = np.pi/4
dvl_v_x = (dvl_VE*np.cos(angle) + dvl_VN*np.sin(angle))*np.cos(YAW)
dvl_v_y = (dvl_VN*np.sin(angle) + dvl_VE*np.cos(angle))*np.sin(YAW)
dvl_v_z = dvl_VZ


if __name__ == '__main__':
    def distance_to_bottom(xy,mnt):
        d_mnt, indices = kd_tree.query(xy)  #Utilise KDTree pour calculer les distances
        Z = mnt[indices,2] # Récupère les altitudes des points les plus proches
        return d_mnt, Z
    x_gps, y_gps = coord2cart((LAT, LON))

    import matplotlib.pyplot as plt
    T = (T - T[0])/60
    dvl_T = (dvl_T - dvl_T[0])/60
    MBES_T = (MBES_T - MBES_T[0])/60
    mean_dvlR = (dvl_BM1R + dvl_BM2R + dvl_BM3R + dvl_BM4R)/4
    print(f"the offset with dt = {dt_br} for the DVL/MNT is : {np.mean(mean_dvlR) - np.mean(distance_to_bottom(np.column_stack((x_gps,y_gps)),MNT)[1].squeeze())}")
    print(f"the offset with dt = {dt_br} for the MBES/MNT is : {np.mean(MBES_Z) - np.mean(distance_to_bottom(np.column_stack((x_gps,y_gps)),MNT)[1].squeeze())}")
    def display_range():
        #######################################
        plt.figure()
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        ax5 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        # ax1 = plt.subplot2grid((2, 3), (0, 0))
        # ax2 = plt.subplot2grid((2, 3), (0, 1))
        # ax3 = plt.subplot2grid((2, 3), (1, 0))
        # ax4 = plt.subplot2grid((2, 3), (1, 1))
        # ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

        mask1 = np.abs(np.diff(MBES_Z)) > 0.001*np.std(MBES_Z)

        d_bottom_mnt = distance_to_bottom(np.column_stack((x_gps,y_gps)),MNT)[1].squeeze()

        mean_dvlR = (dvl_BM1R + dvl_BM2R + dvl_BM3R + dvl_BM4R)/4

        ax1.plot(dvl_T, dvl_BM1R - 115.57108493670452, label = "dvl_BM1R", color = 'red')
        ax1.plot((MBES_T[1:,])[~mask1], (MBES_Z[1:,])[~mask1] - 117.6152233539319, label="MBES_Z")
        ax1.plot(T, d_bottom_mnt, label = "d_mnt")
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel("Time [min]")
        ax1.set_ylabel("Range [m]")
        ax1.set_title("dvl_BM1R")

        # ax1.plot(dvl_T, mean_dvlR - np.mean(mean_dvlR), label = "dvl_R - mean(dvl_R)", color = 'red')
        # ax1.plot((MBES_T[1:,])[~mask1], (MBES_Z[1:,])[~mask1] - np.mean((MBES_Z[1:,])[~mask1]), label="MBES_Z - mean(MBES_Z)")
        # ax1.plot(T, d_bottom_mnt - np.mean(d_bottom_mnt), label = "d_z_mnt - mean(d_z_mnt)")
        # ax1.legend()
        # ax1.grid()
        # ax1.set_xlabel("Time [min]")
        # ax1.set_ylabel("Range [m]")
        # ax1.set_title("Range from mean range")


        # ax2.plot(dvl_T, dvl_BM2R - np.mean(dvl_BM2R), label = "dvl_BM2R", color = 'red')
        # ax2.plot((MBES_T[1:,])[~mask1], (MBES_Z[1:,])[~mask1] - np.mean((MBES_Z[1:,])[~mask1]), label="MBES_Z")
        # ax2.plot(T, d_bottom_mnt, label = "d_mnt")
        # ax2.legend()
        # ax2.grid()
        # ax2.set_xlabel("Time [min]")
        # ax2.set_ylabel("Range [m]")
        # ax2.set_title("dvl_BM2R")
        #
        # ax3.plot(dvl_T, dvl_BM3R - np.mean(dvl_BM3R), label = "dvl_BM3R", color = 'red')
        # ax3.plot((MBES_T[1:,])[~mask1], (MBES_Z[1:,])[~mask1] - np.mean((MBES_Z[1:,])[~mask1]), label="MBES_Z")
        # ax3.plot(T, d_bottom_mnt, label = "d_mnt")
        # ax3.legend()
        # ax3.grid()
        # ax3.set_xlabel("Time [min]")
        # ax3.set_ylabel("Range [m]")
        # ax3.set_title("dvl_BM3R")
        #
        # ax4.plot(dvl_T, dvl_BM4R - np.mean(dvl_BM4R), label = "dvl_BM4R", color = 'red')
        # ax4.plot((MBES_T[1:,])[~mask1], (MBES_Z[1:,])[~mask1] - np.mean((MBES_Z[1:,])[~mask1]), label="MBES_Z")
        # ax4.plot(T, d_bottom_mnt, label = "d_mnt")
        # ax4.legend()
        # ax4.grid()
        # ax4.set_xlabel("Time [min]")
        # ax4.set_ylabel("Range [m]")
        # ax4.set_title("dvl_BM4R")

        ax5.scatter(T[:-1,], np.diff(distance_to_bottom(np.column_stack((x_gps,y_gps)),MNT)[1].squeeze()), label = "d_mnt", s = 1)
        mean_dvlR = (dvl_BM1R + dvl_BM2R + dvl_BM3R + dvl_BM4R)/4
        ax5.scatter(dvl_T[:-1,], np.diff(mean_dvlR), label = "mean dvl_BMR", color = 'orange', s = 1)
        ax5.scatter(MBES_T[:-1,], np.diff(MBES_Z), label="MBES_Z", color = 'gray', s = 0.5)
        ax5.legend()
        ax5.grid()
        ax5.set_xlabel("Time [min]")
        ax5.set_ylabel("Range [m]")
        ax5.set_title("Mean of dvl range v/s MBES")
        # ax5.scatter(MBES_T[1:-1,][~mask1], np.diff(MBES_Z[1:,][~mask1]), label="MBES_Z")

        # Calculate the error between the DVL and the MNT
        dvl_error = np.diff(mean_dvlR) - np.diff(distance_to_bottom(np.column_stack((x_gps,y_gps)),MNT)[1].squeeze())
        # Calculate the error between the MBES and the MNT
        mbes_error = np.diff(MBES_Z) - np.diff(distance_to_bottom(np.column_stack((x_gps,y_gps)),MNT)[1].squeeze())

        plt.figure()
        plt.scatter(MBES_T[:-1], mbes_error, label = "mbes_error", color = 'green', s = 1)
        plt.scatter(T[:-1], dvl_error, label = "dvl_error", color = 'red', s = 1)
        plt.legend()
        plt.grid()
        plt.xlabel("Time [min]")
        plt.ylabel("Error on measuremnts [m]")
        plt.title("Error of dvl range v/s MNT and mbes range v/s MNT")
        plt.plot()
    display_range()
    def display_speed():
        ##################################################
        plt.figure()

        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        ax4 = plt.subplot2grid((2, 3), (1, 1))
        ax5 = plt.subplot2grid((2, 3), (0, 2))
        ax6 = plt.subplot2grid((2, 3), (1, 2))


        ax1.plot(dvl_T, dvl_v_x, label="dvl")
        ax1.plot(T, (V_X), label = "ins")
        # B = ACC_X[:ACC_X.shape[0,]-1,]
        # C = np.insert(B, 0, 0)
        # print(np.cumsum(ACC_X)*0.05 + V_X[0,])
        # ax1.plot(T, np.cumsum(ACC_X)*0.05 + V_X[0,], label = "dt*acc")
        ax1.legend()
        ax1.set_title("VE")
        ax1.grid()
        ax1.set_xlabel("Time [min]")
        ax1.set_ylabel("Speed [m/s]")


        ax2.plot(dvl_T, dvl_v_y, label="dvl")
        ax2.plot(T, (V_Y), label = "ins")
        # ax2.plot(T, np.cumsum(ACC_Y)*0.05 + dvl_v_y[0,], label = "dt*acc")
        ax2.legend()
        ax2.set_title("VN")
        ax2.grid()
        ax2.set_xlabel("Time [min]")
        ax2.set_ylabel("Speed [m/s]")

        ax3.scatter(dvl_T, (dvl_VZ), label="dvl")
        ax3.scatter(T, (V_Z), label = "ins")
        # ax3.plot(T, np.cumsum(ACC_Z)*0.05 + dvl_v_z[0,], label = "dt*acc")
        ax3.legend()
        ax3.set_title("VZ")
        ax3.grid()
        ax3.set_xlabel("Time [min]")
        ax3.set_ylabel("Speed [m/s]")

        ax4.plot(dvl_T, np.sqrt(dvl_v_x**2 + dvl_v_y**2), label="dvl")
        ax4.plot(T, np.sqrt(V_X**2 + V_Y**2), label = "ins")
        ax4.plot(T, 0.05**2*np.sqrt(ACC_X**2 + ACC_Y**2), label = "ins")
        ax4.set_title("||V_xy||")
        ax4.legend()
        ax4.grid()
        ax4.set_xlabel("Time [min]")
        ax4.set_ylabel("Speed [m/s]")

        ax5.scatter(dvl_T, dvl_VSTD/10, label = "error speed on dvl", s = 0.5)
        ax5.scatter(T, V_X_STD, label = "error speed on ins on X", s = 0.5)
        ax5.scatter(T, V_Y_STD, label = "error speed on ins on Y", s = 0.5)
        ax5.scatter(T, V_Z_STD, label = "error speed on ins on Z", s = 0.5)
        ax5.set_title("Error on speed")
        ax5.legend()
        ax5.grid()
        ax5.set_xlabel("Time [min]")
        ax5.set_ylabel("Error on speed [m/s]")

        ax6.plot(dvl_T, np.arctan2(dvl_v_y,dvl_v_x), label = "angle_vDvL")
        ax6.plot(T, np.arctan2(V_Y,V_X), label = "angle_vINS")
        # ax6.plot(T, YAW, label = f"yaw")
        ax6.legend()
        ax6.grid()
        ax6.set_xlabel("Time [min]")
        ax6.set_ylabel("Error on angle [rad]")
        ax6.set_title("angle of speed")
    display_speed()
    def display_acc():
        plt.figure()
        plt.plot(T, ACC_X, label = "acc_x")
        plt.plot(T, ACC_Y, label = "acc_y")
        plt.plot(T, ACC_Z, label = "acc_z")
        plt.grid()
        plt.xlabel("time [min]")
        plt.ylabel("acc [m/s2]")
        plt.legend()
    # display_acc()
    plt.show()

    # plt.figure()
    # plt.plot(T, PITCH, label = "pitch")
    # plt.plot(T, ROLL, label = "roll")
    # plt.plot(T, YAW, label = "yaw")
    # plt.legend()
    # plt.show()

    # plt.figure()
    # # V = np.sqrt(V_X**2 + V_Y**2)
    # plt.plot(T, V_X, label = "ins", color = 'green')
    # plt.plot(T, dvl_v_x, label = "dvl", color = 'red')
    # ERR = (V_X - dvl_v_x)
    # kp = 10
    # ki = 0
    # kd = 1
    # plt.scatter(T[:-1,], kp*ERR[:-1,] + ki*np.cumsum(ERR)[:-1,]*dt + kd*np.diff(ERR)/dt, label = "pid", s = 0.5, color = 'blue')
    # plt.title("speed")
    # plt.xlabel("t [min]")
    # plt.ylabel("speed [m/s]")
    # plt.legend()
    # plt.show()
