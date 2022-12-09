import numpy as np 



def interpolation(lat, lon, mnt):
	lat_mnt = mnt[:,0]
	lon_mnt = mtn[:,1]

	min_dist = 1e5
	for i in range(len(lat_mnt)):
		if np.sqrt((lat_mnt[i]-lat)**2+(lon_mnt[i]-lon)**2)<=min_dist:
			min_dist = np.sqrt((lat_mnt[i]-lat)**2+(lon_mnt[i]-lon)**2)
			i_pos = i

	if lat_mnt[i_pos]-lat<=0 and lon_mnt[i_pos]-lon<=0:
		return (mnt[i_pos, 3] + mnt[i_pos+1, 3])/2

	elif lat_mnt[i_pos]-lat>=0 and lon_mnt[i_pos]-lon>=0:
		return (mnt[i_pos-1, 3] + mnt[i_pos, 3])/2

	else: 
		return mnt[i_pos, 3]



if __name__ == "__main__":
	# interpolation()
	print("J'interpole mamène")