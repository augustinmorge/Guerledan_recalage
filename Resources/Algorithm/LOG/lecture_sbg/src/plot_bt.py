import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__" :
	filename="../beuteu.txt"
	f=open(filename,"r")
	data=f.readlines()
	bt_range=[]
	bt_range_off=[]
	time=[]
	for i in range(len(data)) :
		bt_range.append(int(data[i])/100)
		bt_range_off.append(int(data[i])/100-117.61)
		# time.append(i*)
	bt_range=np.array(bt_range)
	bt_range_off=np.array(bt_range_off)
	plt.figure()
	plt.plot(bt_range)
	plt.plot(bt_range_off)
	plt.show()