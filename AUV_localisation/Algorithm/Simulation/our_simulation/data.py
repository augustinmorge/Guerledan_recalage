import numpy as np
import os

def data(data_file="sbgCenterExport.txt") :
	data = open(data_file,'r')
	data.readline()
	data.readline()

	n=1076638
	time=np.zeros(n)
	velocity=np.zeros((n,3))
	gps=np.zeros((n,3))

	t_offset=2700.749
	for k in range(n):
		line=data.readline().split('\t')
		time[k]=float(line[0])-t_offset
		velocity[k,0]=float(line[3])
		velocity[k,1]=float(line[4])
		velocity[k,2]=float(line[5])
		gps[k,0]=float(line[6])
		gps[k,1]=float(line[7])
		gps[k,2]=float(line[8])
	return time,velocity,gps


salut,yo,bonjour=data()
print(bonjour)