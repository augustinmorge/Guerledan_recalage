# Compute the number of times each sample should be replicated
Nm = np.floor(N * wm)

# Compute the indices of the samples that will be replicated
indices = np.nonzero(Nm)[0]

# Compute the number of replicated samples
Nt = np.sum(Nm)

# Replicate each sample the appropriate number of times
new_samples_deterministic = []
for i in indices:
    new_samples_deterministic.append(xm[0][i] * np.ones((Nm[i], 1)))
    new_samples_deterministic.append(xm[1][i] * np.ones((Nm[i], 1)))

# Concatenate the replicated samples into a single array
new_samples_deterministic = [np.concatenate(x, axis=0) for x in new_samples_deterministic]


# # Compute replication
# Nm = np.floor(N * wm)
#
# # Store sample to be used for replication
# Nm = Nm.astype(int)
# replication_samples = [xm, Nm]
#
# # Replicate samples
# new_samples_deterministic = []
# Nt = np.sum(Nm)
# val_x = np.zeros((Nt,1))
# val_y = np.zeros((Nt,1))
# import time
# for i in range(Nm.shape[0]):
#     idx_pre = np.sum(Nm[:i,])
#     idx_now = np.sum(Nm[:i+1,])
#     if idx_now - idx_pre != 0:
#         val_x[idx_pre:idx_now,] = xm[0][i,0]*np.ones((Nm[i,0], 1))
#         val_y[idx_pre:idx_now,] = xm[1][i,0]*np.ones((Nm[i,0], 1))
#
# new_samples_deterministic.append(val_x)
# new_samples_deterministic.append(val_y)
