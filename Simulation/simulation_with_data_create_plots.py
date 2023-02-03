#!/usr/bin/env python3
import sys
print("You can run your program adding mbes or dvl or mnt at the end of the arguments")
if len(sys.argv[1:])>=1:
    for arg in sys.argv[1:]:
        if arg == "mnt" or arg == "dvl" or arg == "mbes":
            choice_range_sensor = arg
else:
    choice_range_sensor = str(input("Choose your way to measure the bottom range [mnt/dvl/mbes]: "))
    if choice_range_sensor not in ["mnt","dvl","mbes"]:
        print("You have to select a sensor")
        sys.exit()

from storage.data_import import *
steps = int(input("number of steps between measures ? "))
using_offset = True # str(input("Using offset ? [Y/]")) == "Y"

ct_resampling = 0

import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
from resampler import Resampler
from tqdm import tqdm
file_path = os.path.dirname(os.path.abspath(__file__))
from filter import *

def sawtooth(x):
    return(2*np.arctan(np.tan(x/2)))

def distance_to_bottom(xy,mnt):
    d_mnt, indices = kd_tree.query(xy)  #Utilise KDTree pour calculer les distances
    Z = mnt[indices,2] # Récupère les altitudes des points les plus proches
    return d_mnt, Z

def initialize_particles_uniform(n_particles):
    particles = []
    index = np.random.randint(0, MNT.shape[0], size = n_particles)
    particles = [MNT[index][:,0].reshape(-1,1), MNT[index][:,1].reshape(-1,1)]
    particles = [1/n_particles*np.ones((n_particles,1)), particles]
    return(particles)

def normalize_weights(weighted_samples):
    return [weighted_samples[0] / np.sum(weighted_samples[0]), weighted_samples[1]]

def validate_state(state, d_mnt):
    # # If we are out of the MNT
    # weights = state[0]
    # weights[d_mnt > 1] = 0
    return(state)

def propagate_sample(samples, forward_motion, angular_motion, process_noise):

    # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
    # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
    forward_displacement = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(-1,1)
    # forward_displacement_y = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(n_particles,1)
    angular_motion = np.random.normal(angular_motion, process_noise[1],n_particles).reshape(-1,1)

    # 2. move forward
    samples[1][0] += forward_displacement*np.cos(angular_motion)
    samples[1][1] += forward_displacement*np.sin(angular_motion)

    # Make sure we stay within cyclic world
    return samples

def compute_likelihood(propagated_states, measurements, measurements_noise, beta, z_particules_mnt):
    d_mnt, new_z_particules_mnt = distance_to_bottom(np.hstack((propagated_states[1][0],propagated_states[1][1])),MNT)
    if using_offset : d_mbes_particule = new_z_particules_mnt
    else : d_mbes_particule = new_z_particules_mnt - z_particules_mnt

    # Map difference true and expected distance measurement to probability
    distance = np.abs(d_mbes_particule-measurements)

    if measurements_noise[0] == None:
        p_z_given_x_distance = np.exp(-beta*distance)
    else:
        p_z_given_x_distance = np.exp(-beta*distance/(measurements_noise[0]**2))

    # p_z_given_x_distance = 1
    # Return importance weight based on all landmarks
    return d_mnt, p_z_given_x_distance, new_z_particules_mnt

def needs_resampling(resampling_threshold):
    return 1.0 / np.max(particles[0]) < resampling_threshold


def update(robot_forward_motion, robot_angular_motion, measurements, \
            measurements_noise, process_noise, particles,resampling_threshold,\
            resampler, beta, z_particules_mnt):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise)

    # Compute current particle's weight
    d_mnt, p, z_particules_mnt = compute_likelihood(propagated_states, measurements, measurements_noise, beta, z_particules_mnt)

    particules = validate_state(propagated_states, d_mnt)

    # Update the probability of the particle
    propagated_states[0] = particles[0] * p

    # Update particles
    particles = normalize_weights(propagated_states)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        global ct_resampling
        ct_resampling += 1
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles, z_particules_mnt)

def get_average_state(particles):

    # Compute weighted average
    avg_x = np.sum(particles[0]*particles[1][0]) / np.sum(particles[0])
    avg_y = np.sum(particles[0]*particles[1][1]) / np.sum(particles[0])

    return [avg_x, avg_y]

# Init range sensor
if choice_range_sensor == "mnt":
    x_gps, y_gps = coord2cart((LAT[0,],LON[0,])).flatten()
    d_mnt, previous_measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
elif choice_range_sensor == "dvl":
    previous_measurements = (dvl_BM1R[0,] + dvl_BM2R[0,] + dvl_BM3R[0,] + dvl_BM4R[0,])/4 #range__Z[0,]
else:
    previous_measurements = MBES_Z[0,]

def f_measurements(i, previous_measurements):
    if choice_range_sensor == "mnt":
        x_gps, y_gps = coord2cart((LAT[i,],LON[i,])).flatten()
        d_mnt, measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
        return measurements-previous_measurements, measurements, None #d_mnt
        # return measurements, measurements, d_mnt
    elif choice_range_sensor == "dvl":
        mean_range_dvl = (dvl_BM1R[i,] + dvl_BM2R[i,] + dvl_BM3R[i,] + dvl_BM4R[i,])/4
        measurements = mean_range_dvl - previous_measurements #117.61492204 #
        return measurements, mean_range_dvl, None
    else:
        measurements = MBES_Z[i,] - previous_measurements #117.61492204 #
        return measurements, MBES_Z[i,], None

def f_measurements_offset(i):
    if choice_range_sensor == "mnt":
        x_gps, y_gps = coord2cart((LAT[i,],LON[i,])).flatten()
        d_mnt, measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
        return measurements, d_mnt
    elif choice_range_sensor == "dvl":
        mean_range_dvl = (dvl_BM1R[i,] + dvl_BM2R[i,] + dvl_BM3R[i,] + dvl_BM4R[i,])/4
        measurements = mean_range_dvl - 115.57149562238688
        return measurements, None
    else:
        measurements = MBES_Z[i,] - 117.61544705067318
        return measurements, None

def test_diverge(ERR, err_max=1000):
    if ERR[-1] > err_max: #Si l'erreur est de plus de 500m il y a un probleme
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Divergence of the algorithm")
        print(f"dt = {dt}")
        print(f"process_noise = {process_noise}")
        print(f"measurements_noise = {measurements_noise}")
        print(f"V = {v_x, v_y, v_z}")
        print(f"pos_std = {lat_std, lon_std}")
        print(f"speed_std = {v_std}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return True #Alors on arrete
    return(False)


if __name__ == '__main__':
    print("~~~Start of the algorithm~~~")

    x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0,:]), np.min(coord2cart((LAT, LON))[1,:])
    x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0,:]), np.max(coord2cart((LAT, LON))[1,:])
    # bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]

    idx_ti = 0 + steps
    idx_tf =  dvl_T.shape[0]

    dt = dvl_T[steps,] - dvl_T[0,]
    tini = dvl_T[idx_ti,]
    tf = dvl_T[idx_tf-1,]

    v_x = dvl_VE[idx_ti,]
    v_y = dvl_VN[idx_ti,]
    v_z = dvl_VZ[idx_ti,]
    v_std = dvl_VSTD[idx_ti,]

    lat = LAT[idx_ti,]
    lon = LON[idx_ti,]
    lat_std = LAT_STD[idx_ti,]
    lon_std = LON_STD[idx_ti,]

    fig, ax = plt.subplots()
    BETA = [1/1000, 1/500, 1/300, 1/100, 5/100, 1/10, 1/5, 1/2, 1]
    # filter_speed = Low_pass_filter(0.4, np.array([dvl_v_x[0,], dvl_v_y[0,]]))

    # N_PARTICULES = [j for j in range(500,5500,500)]
    N_PARTICULES = [2000]
    for n_particles in N_PARTICULES:
        ERROR = []
        #For the update
        resampler = Resampler()
        resampling_threshold = 0.5*n_particles
        particles = initialize_particles_uniform(n_particles)
        _, z_particules_mnt = distance_to_bottom(np.hstack((particles[1][0],particles[1][1])),MNT)
        print("n_particles = ",n_particles)
        for k in tqdm(range(len(BETA))):
            beta = BETA[k]
            TIME = []; BAR = []; SPEED = []; ERR = []
            STD_X = []; STD_Y = []
            MEASUREMENTS = []
            # r = tqdm(range(idx_ti,idx_tf,steps))
            r = (range(idx_ti,idx_tf,steps))
            for i in r:

                """Set data"""
                t = dvl_T[i,]
                yaw = YAW[i,]
                yaw_std = YAW_STD[i,]
                # v_x, v_y = filter_speed.low_pass_next(np.array([dvl_v_x[i,], dvl_v_y[i,]])).flatten()
                v_x = dvl_v_x[i,]
                v_y = dvl_v_y[i,]

                # v_x = V_X[i,]
                # v_y = V_Y[i,]
                # v_std = np.sqrt(V_X_STD[i,]**2+V_Y_STD[i,]**2)
                v_std = dvl_VSTD[i,]/10
                if using_offset : measurements, meas_model_distance_std = f_measurements_offset(i)
                else: measurements, previous_measurements, meas_model_distance_std = f_measurements(i, previous_measurements)

                """Processing the motion of the robot """
                robot_forward_motion =  dt*np.sqrt(v_x**2 + v_y**2)
                robot_angular_motion = yaw

                """ Processing error on measures"""
                measurements_noise = [meas_model_distance_std] ### Attention, std est en mètres !

                """ Processing error on algorithm"""
                motion_model_forward_std = steps*np.abs(v_std)
                # motion_model_turn_std = np.abs(sawtooth(np.arctan2((v_x + np.sign(v_x)*v_std),(v_y)) - np.arctan2((v_x),(v_y+np.sign(v_y)*v_std))))
                motion_model_turn_std = yaw_std
                process_noise = [motion_model_forward_std, motion_model_turn_std]

                """Process the update"""
                t0 = time.time()
                particles, z_particules_mnt = update(robot_forward_motion, robot_angular_motion, measurements,\
                                                       measurements_noise, process_noise, particles,\
                                                        resampling_threshold, resampler, beta,\
                                                        z_particules_mnt)

                #Add variables useful to display graphs at the end of the program
                TIME.append(t)

                x_gps, y_gps = coord2cart((LAT[i,],LON[i,])).flatten()
                ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))


            ERROR.append(np.mean(np.array(ERR)))
            for err in ERROR:
                if err == np.nan:
                    err = 1000
        print(BETA, ERROR)
        plt.figure()
        plt.scatter(np.log(BETA), ERROR)
        plt.xlabel("log(beta)")
        plt.ylabel("error")
        plt.title("Test error with beta")
        plt.grid()
        plt.savefig("./imgs/test_beta_2/Test_with_nb_part={}.png".format(n_particles,beta))

print("~~~End of the algorithm~~~")
