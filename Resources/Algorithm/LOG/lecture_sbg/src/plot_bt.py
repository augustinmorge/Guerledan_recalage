import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__" :
	filename="../dvl_final.txt"
	f=open(filename,"r")
	data=f.readlines()
	b1t_range=[]
	b2t_range=[]
	b3t_range=[]
	b4t_range=[]
	bt_range_off=[]
	time=[]
	error_vel=[]
	for i in range(1,len(data)) :
		data_=data[i].split(",")
		err=int(data_[-1].split("\n")[0])
		if abs(err) < 32000 :
			error_vel.append(err)
		else :
			error_vel.append(26)
		if int(data_[7])!=0 :
			b1t_range.append(int(data_[7])/100-117.61)
			b1_ex=int(data_[7])/100-117.61
		else :
			b1t_range.append(b1_ex)
		if int(data_[8])!=0 :
			b2t_range.append(int(data_[8])/100-117.61)
			b2_ex=int(data_[8])/100-117.61
		else :
			b2t_range.append(b2_ex)
		if int(data_[9])!=0 :
			b3t_range.append(int(data_[9])/100-117.61)
			b3_ex=int(data_[9])/100-117.61
		else :
			b3t_range.append(b3_ex)
		if int(data_[10])!=0 :
			b4t_range.append(int(data_[10])/100-117.61)
			b4_ex=int(data_[10])/100-117.61
		else :
			b4t_range.append(b4_ex)
		# bt_range_off.append(int(data_[7])/100-117.61)
	b1t_range=np.array(b1t_range)
	b2t_range=np.array(b2t_range)
	b3t_range=np.array(b3t_range)
	b4t_range=np.array(b4t_range)
	# bt_range_off=np.array(bt_range_off)
	plt.figure()
	plt.plot(b1t_range)
	plt.plot(b2t_range)
	plt.plot(b3t_range)
	plt.plot(b4t_range)
	plt.legend(("Beam 1","Beam 2","Beam 3","Beam 4"))
	# plt.plot(bt_range_off)
	plt.figure()
	plt.plot(error_vel)

	plt.show()