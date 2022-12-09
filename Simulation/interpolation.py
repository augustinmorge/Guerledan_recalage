import numpy as np 
import time
from scipy.spatial import cKDTree


def interpolation_float(lat, lon, mnt):
	lat_mnt = mnt[:,0]
	lon_mnt = mnt[:,1]
	points = np.vstack((lat_mnt, lon_mnt)).T

	point = np.array([lat, lon])
	distances = np.linalg.norm(points - point, axis=1)
	i_pos = np.argmin(distances)

	if lat_mnt[i_pos]-lat<=0 and lon_mnt[i_pos]-lon<=0:
		return (mnt[i_pos][2] + mnt[i_pos+1][2])/2

	elif lat_mnt[i_pos]-lat>=0 and lon_mnt[i_pos]-lon>=0:
		return (mnt[i_pos-1][2] + mnt[i_pos][2])/2

	else: 
		return mnt[i_pos][2]

def interpolation(lat, lon, mnt):
	lat_mnt = mnt[:,0]
	lon_mnt = mnt[:,1]
	points = np.vstack((lat_mnt, lon_mnt)).T

	if type(lat) == float:
		return interpolation_float(lat ,lon, mnt)
	
	else:
		print("Pour le nuage de particules")
		particules = np.vstack((lat, lon)).T
		kd_tree = cKDTree(points)
		for particule in particules:
			dist, i_pos = kd_tree.query(particule)
		return(mnt[i_pos][2])


if __name__ == "__main__":
	print("\nLoading data ...")
	AimeNT = np.loadtxt("./resources/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)
	MNT = []
	for i in AimeNT:
		MNT.append(i.split(','))
		MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])] 
	MNT = np.array(MNT)
	print("Data loaded, ready to interpole !\n")

	lat, lon = np.arange(0,1000,1), np.arange(0,1000,1)
	t0 = time.time()
	interpolation(48.1989495, -3.0148023, MNT)
	print("Temps avec numpy : ", time.time()-t0)
	t1 = time.time()
	interpolation(lat, lon, MNT)
	print("Temps avec cKDTree : ", time.time()-t1)
