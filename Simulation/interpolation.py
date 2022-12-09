import numpy as np 



def interpolation(lat, lon, mnt):
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

def interpolation2(lat, lon, mnt):
	lat_mnt = mnt[:,0]
	lon_mnt = mnt[:,1]
	points = np.vstack((lat_mnt, lon_mnt)).T

	min_dist = 1e5
	for i in range(len(lat_mnt)):
		if np.sqrt((lat_mnt[i]-lat)**2+(lon_mnt[i]-lon)**2)<=min_dist:
			min_dist = np.sqrt((lat_mnt[i]-lat)**2+(lon_mnt[i]-lon)**2)
			i_pos = i

	if lat_mnt[i_pos]-lat<=0 and lon_mnt[i_pos]-lon<=0:
		return (mnt[i_pos][2] + mnt[i_pos+1][2])/2

	elif lat_mnt[i_pos]-lat>=0 and lon_mnt[i_pos]-lon>=0:
		return (mnt[i_pos-1][2] + mnt[i_pos][2])/2

	else: 
		return mnt[i_pos][2]



if __name__ == "__main__":
	AimeNT = np.loadtxt("./resources/guerledan_EDF_2013-06_MNT1m.tiff.txt", dtype = str)
	MNT = []
	for i in AimeNT:
		MNT.append(i.split(','))
		MNT[-1] = [np.float64(MNT[-1][0]), np.float64(MNT[-1][1]), np.float64(MNT[-1][2]+'.'+MNT[-1][3])] 
	MNT = np.array(MNT)
	print(interpolation(48.1989495, -3.0148023, MNT))
	# print(interpolation2(48.1989495, -3.0148023, MNT))

