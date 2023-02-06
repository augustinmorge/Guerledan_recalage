#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 1:
    print("Please add the file at the end of the command")
    sys.exit()

for arg in sys.argv[1:]:
    if arg[-4:] != ".txt":
        print("Please add the file at the end of the command")
        sys.exit()
    else:
        filename = arg

file = open(str(filename), "r")
BETA = []; ERR = []
for line in file.readlines():
    err, beta = line.split(";")
    BETA.append(beta)
    ERR.append(err)

BETA = np.float64(BETA)
ERR = np.float64(ERR)

plt.figure()
plt.scatter(np.log(BETA)/np.log(10), ERR)
err_min = np.min(ERR)
for i in range(ERR.shape[0]):
    if ERR[i] == err_min:
        plt.scatter(np.log(BETA[i,])/np.log(10), ERR[i,], color = 'red',\
         label = 'err_min for beta = {}'.format(BETA[i,]))

# for i in range(len(ERROR)):
#     plt.scatter(np.log(BETA[i])/np.log(10), ERROR[i], label = str(int(BETA[i]*10000000)/10000000))
plt.xlabel("log(beta)")
plt.ylabel("error")
plt.title("Test error with beta")
plt.legend()
plt.grid()

f = plt.gcf()
dpi = f.get_dpi()
h, w = f.get_size_inches()
f.set_size_inches(h*4, w*4)
f.savefig("{}.png".format(str(filename)))
file.close()
plt.show()
