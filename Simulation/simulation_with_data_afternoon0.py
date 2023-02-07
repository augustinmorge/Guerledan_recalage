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

from storage_afternoon.data_import_new import *
n_particles = int(input("Number of particles: "))
steps = int(input("number of steps between measures ? "))
bool_display = (str(input("Display the particles ? [Y/]"))=="Y")
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
        p_z_given_x_distance = np.exp(-beta*distance**2)
    else:
        p_z_given_x_distance = np.exp(-beta*distance**2/(measurements_noise[0]**2))

    # p_z_given_x_distance = 1
    # Return importance weight based on all landmarks
    return d_mnt, p_z_given_x_distance, new_z_particules_mnt

def needs_resampling(resampling_threshold):
    # return 1.0 / np.max(particles[0]) < resampling_threshold
    return(1.0/np.sum(particles[0]**2) < resampling_threshold)

def update(robot_forward_motion, robot_angular_motion, measurements, \
            measurements_noise, process_noise, particles,resampling_threshold,\
            resampler, beta, z_particules_mnt):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise)

    # Compute current particle's weight
    d_mnt, p, z_particules_mnt = compute_likelihood(propagated_states, measurements, measurements_noise, beta, z_particules_mnt)

    particles = validate_state(propagated_states, d_mnt)

    # Update the probability of the particle
    propagated_states[0] = particles[0] * p

    # Update particles
    particles = normalize_weights(propagated_states)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        global ct_resampling
        ct_resampling += 1
        particles = resampler.resample(particles, n_particles)

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
        return measurements, None #d_mnt
    elif choice_range_sensor == "dvl":
        mean_range_dvl = (dvl_BM1R[i,] + dvl_BM2R[i,] + dvl_BM3R[i,] + dvl_BM4R[i,])/4
        measurements = mean_range_dvl - 116.48084912914656
        return measurements, None
    else:
        measurements = MBES_Z[i,] - 117.67756491403492
        return measurements, None

def test_diverge(ERR, err_max=1000):
    if ERR[-1] > err_max: #Si l'erreur est de plus de 500m il y a un probleme
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Divergence of the algorithm")
        print(f"dt = {dt}")
        print(f"process_noise = {process_noise}")
        print(f"measurements_noise = {measurements_noise}")
        print(f"V = {v_x, v_y}")
        print(f"yaw_std = {yaw_std}")
        print(f"speed_std = {v_std}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return True #Alors on arrete
    return(False)


if __name__ == '__main__':
    print("~~~Start of the algorithm~~~")

    x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0,:]), np.min(coord2cart((LAT, LON))[1,:])
    x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0,:]), np.max(coord2cart((LAT, LON))[1,:])
    # bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
    particles = initialize_particles_uniform(n_particles)

    _, z_particules_mnt = distance_to_bottom(np.hstack((particles[1][0],particles[1][1])),MNT)

    #For the update
    resampler = Resampler()
    resampling_threshold = 2/3*n_particles

    idx_ti = 0
    idx_tf =  dvl_T.shape[0]

    dt = dvl_T[steps,] - dvl_T[0,]
    tini = dvl_T[idx_ti,]

    if bool_display:
        """ Création des isobates """
        plt.ion()


        x = np.linspace(-np.min(LON), 120, 100)
        y = np.linspace(-120, 120, 100)
        X, Y = np.meshgrid(x, y)

        print("Processing..")
        r = range(idx_ti,idx_tf,steps)


    else : r = tqdm(range(idx_ti,idx_tf,steps))

    fig, ax = plt.subplots()
    TIME = []; BAR = []; SPEED = []; ERR = []
    STD_X = []; STD_Y = []
    # beta = 5/100
    # beta = 1/10
    beta = 10**(-1.37)
    filter_lpf_speed = Low_pass_filter(1., np.array([dvl_v_x[0,], dvl_v_y[0,]]))
    filter_lpf_mbes = Low_pass_filter(0.1, np.array([MBES_Z[0,]]))
    MEASUREMENTS = []
    for i in r:

        """Set data"""
        t = dvl_T[i,]
        yaw = YAW[i,]
        yaw_std = YAW_STD[i,]
        v_x, v_y = filter_lpf_speed.low_pass_next(np.array([dvl_v_x[i,], dvl_v_y[i,]])).flatten()
        # v_std = dvl_VSTD[i,]
        # v_std = 0.4*10*dt_br
        v_std = 0.4*10*dt_br

        if using_offset : measurements, meas_model_distance_std = f_measurements_offset(i)
        else: measurements, previous_measurements, meas_model_distance_std = f_measurements(i, previous_measurements)

        measurements = filter_lpf_mbes.low_pass_next(np.array([MBES_Z[i,]]))
        MEASUREMENTS.append(measurements)

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

        """ Affichage en temps réel """
        if bool_display:
            lat = LAT[i,]
            lon = LON[i,]
            x_gps, y_gps = coord2cart((lat,lon)).flatten()
            ax.cla()
            print("Temps de calcul: ",time.time() - t0)
            t1 = time.time()
            ax.plot(coord2cart((LAT[idx_ti:idx_tf,], LON[idx_ti:idx_tf,]))[0,:], coord2cart((LAT[idx_ti:idx_tf,], LON[idx_ti:idx_tf,]))[1,:])
            ax.set_title("Particle filter with {} particles with z = {}m".format(n_particles, measurements))
            ax.set_xlim([x_gps_min - 100,x_gps_max + 100])
            ax.set_ylim([y_gps_min - 100,y_gps_max + 100])
            ax.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée', s = 100)
            ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles") # Affiche toutes les particules
            bx, by = get_average_state(particles)[0], get_average_state(particles)[1] #barycentre des particules
            ax.scatter(bx, by , color = 'green', label = 'Estimation of particles')

            ax.legend()

            plt.pause(0.00001)
            print("Temps d'affichage: ",time.time()-t1,"\n")

        #Add variables useful to display graphs at the end of the program
        TIME.append(t)

        lat = LAT[i,]
        lon = LON[i,]
        x_gps, y_gps = coord2cart((lat,lon)).flatten()
        ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))
        BAR.append([get_average_state(particles)[0],get_average_state(particles)[1]])
        SPEED.append(np.sqrt(v_x**2 + v_y**2))

        var = np.std(np.column_stack((particles[1][0],particles[1][1])),axis=0)
        STD_X.append(var[0])
        STD_Y.append(var[1])

        #Test if the algorithm diverge and why
        if test_diverge(ERR, 500) : break


    print(f"Resampling used: {ct_resampling} ({ct_resampling/((idx_tf - idx_ti)/steps)*100}%)")
    elapsed_time = time.perf_counter() - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    """ Affichage final """
    TIME = (np.array(TIME) - tini)/60.
    BAR = np.array(BAR)
    LAT, LON = LAT[idx_ti:idx_tf,], LON[idx_ti:idx_tf,]
    STD_X = np.array(STD_X).squeeze()
    STD_Y = np.array(STD_Y).squeeze()
    NORM_STD = np.sqrt(STD_X**2 + STD_Y**2)
    max_std = 1.5*np.mean(NORM_STD)
    masque = NORM_STD > max_std

    plt.suptitle(f"Algorithm with {choice_range_sensor}\n{n_particles} particles; 1/{steps} data log used\nTotal time:{int(elapsed_time)}s")
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 1))

    print("Display the error and the final result..")
    ax1.set_title("Barycentre")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.plot(coord2cart((LAT,LON))[0,:], coord2cart((LAT,LON))[1,:],label='true position',linewidth=0.5,color='k')
    scatter = ax1.scatter(BAR[:,0][~masque], BAR[:,1][~masque], s = 1.2, c = NORM_STD[~masque], cmap='plasma', label='barycentre of the particle')
    cbar = fig.colorbar(scatter, extend='both', ax = ax1)
    cbar.set_label('Ecart type')
    ax1.legend()

    ax2.set_title("Error function.")
    ax2.set_xlabel("time [min]")
    ax2.set_ylabel("error (m)")
    ax2.plot(TIME, ERR, color = 'b', label = 'erreur')
    ERR = np.array(ERR)
    idx_start = int(1/8*TIME.shape[0])
    ax2.plot(TIME, np.mean(ERR)*np.ones(TIME.shape), label = f"mean error from beggining = {np.mean(ERR)}")
    ax2.plot(TIME[idx_start:,], np.mean(ERR[idx_start:,])*np.ones(TIME[idx_start:,].shape), label = f"mean error from convergence = {np.mean(ERR[idx_start:,])}")
    ax2.legend()

    X_gps, Y_gps = coord2cart((LAT, LON))
    d_bottom_mnt = distance_to_bottom(np.column_stack((X_gps,Y_gps)),MNT)[1].squeeze()
    mean_dvlR = (dvl_BM1R + dvl_BM2R + dvl_BM3R + dvl_BM4R)/4

    MEASUREMENTS = np.array(MEASUREMENTS)
    ax3.set_title("Different types of bottom measurements")
    ax3.set_xlabel("Time [min]")
    ax3.set_ylabel("Range [m]")
    ax3.plot(TIME, mean_dvlR - 116.48084912914656, label = "z_dvl")
    ax3.plot(TIME, d_bottom_mnt, label = "z_mnt")
    # ax3.plot(MBES_T, MBES_Z - 117.67756491403492, label = "z_mbes")
    ax3.plot(TIME, MEASUREMENTS - 117.67756491403492, label = "z_mbes")
    ax3.legend()

    ax4.set_title("Speed")
    ax4.set_ylabel("v [m/s]")
    ax4.set_xlabel("t [min]")
    ax4.plot((dvl_T[steps:,] - dvl_T[steps,])/60, np.sqrt(dvl_v_x[steps:,]**2 + dvl_v_y[steps:,]**2), label = "dvl_speed")
    ax4.plot(TIME, SPEED, label = "dvl_speed_filtered")
    ax4.plot((T[steps:,] - T[steps,])/60, np.sqrt(V_X[steps:,]**2 + V_Y[steps:,]**2), label = "ins_speed")
    ax4.legend()

    print("Computing the diagrams..")

    plt.show()
    if bool_display:plt.pause(100)


print("~~~End of the algorithm~~~")