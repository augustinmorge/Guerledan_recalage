#!/usr/bin/env python3

n_particles = int(input("Number of particles: "))
steps = int(input("number of steps between measures ? "))
bool_display = (str(input("Display the particles ? [Y/]"))=="Y")

import time
T_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
import copy
from resampler import Resampler
from storage.data_import import *
import sys
from tqdm import tqdm
import pyproj
file_path = os.path.dirname(os.path.abspath(__file__))

# Définit les coordonnées de référence
wpt_ponton = (48.1989495, -3.0148023)

def coord2cart(coords,coords_ref=wpt_ponton):
    R = 6372800
    ly,lx = coords
    lym,lxm = coords_ref
    x_tilde = R * np.cos(ly*np.pi/180)*(lx-lxm)*np.pi/180
    y_tilde = R * (ly-lym)*np.pi/180
    return np.array([x_tilde,y_tilde])

def distance_to_bottom(xy,mnt):
    point = np.vstack((xy[:,0], xy[:,1])).T

    # Utilise KDTree pour calculer les distances
    d_mnt, indices = kd_tree.query(point)

    # Récupère les altitudes des points les plus proches
    Z = mnt[indices,2]

    return d_mnt, Z

def initialize_particles_uniform(n_particles, bounds):
    weight = 1/n_particles
    particles = [weight*np.ones((n_particles,1)),[ \
                           np.random.uniform(bounds[0][0], bounds[0][1], n_particles).reshape(-1,1),
                           np.random.uniform(bounds[1][0], bounds[1][1], n_particles).reshape(-1,1)]]
    return(particles)

def get_max_weight(particles):
    return np.max(weighted_sample[0])

def normalize_weights(weighted_samples):

    # Check if weights are non-zero
    if np.sum(weighted_samples[0]) < 1e-15:
        sum_weights = np.sum(weighted_samples[0])
        print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

        # Set uniform weights
        return [1.0 / weighted_samples[0].shape[0]*np.ones((weighted_samples[0].shape[0],1)), weighted_samples[1]]

    # Return normalized weights
    return [weighted_samples[0] / np.sum(weighted_samples[0]), weighted_samples[1]]

def validate_state(state, bounds, d_mnt):
    x_min, x_max = bounds[0][0] - 10., bounds[0][1] + 10.
    y_min, y_max = bounds[1][0] - 10., bounds[1][1] + 10.

    weights = state[0]
    coords  = state[1]
    weights[(coords[0] < x_min) | (coords[0] > x_max) | (coords[1] < y_min) | (coords[1] > y_max)] = 0
    weights[d_mnt > 2] = 0 # If we are out of the MNT
    if np.sum(weights) == 0: sys.exit()
    return(state)

def propagate_sample(samples, forward_motion, angular_motion, process_noise, bounds):

    # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
    # propagated_samples = copy.deepcopy(samples)
    propagated_samples = samples
    # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
    forward_displacement = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(-1,1)
    # forward_displacement_y = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(n_particles,1)
    angular_motion = np.random.normal(angular_motion, process_noise[1],n_particles).reshape(-1,1)

    # 2. move forward
    propagated_samples[1][0] += forward_displacement*np.cos(angular_motion)
    propagated_samples[1][1] += forward_displacement*np.sin(angular_motion)

    # Make sure we stay within cyclic world
    return propagated_samples

def compute_likelihood(samples, measurements_noise, beta):
    d_mnt, z_mbes_particule = distance_to_bottom(np.hstack((samples[1][0],samples[1][1])),MNT)

    # Calculated with the beams
    distance = 0
    for i in range(-5,6,1):
        for d_beam in range(-10,11,1):
            beams_x = (samples[1][0] + i) + d_beam*np.cos(robot_angular_motion)
            beams_y = (samples[1][1] + i) + d_beam*np.sin(robot_angular_motion)
            _, z_mbes_particule_beam = distance_to_bottom(np.hstack((beams_x,beams_y)),MNT)
            _, measurements = distance_to_bottom(np.array([[(x_gps+i)+d_beam*np.cos(robot_angular_motion),\
                                                            (y_gps+i)+d_beam*np.sin(robot_angular_motion)]]),\
                                                 MNT)
            distance += (z_mbes_particule_beam - measurements)**2
    distance = np.sqrt(distance)
    p_z_given_x_distance = np.exp(-beta*distance/(2*measurements_noise[0]**2))

    # Return importance weight based on all landmarks
    return d_mnt, p_z_given_x_distance

def needs_resampling(resampling_threshold):

    max_weight = np.max(particles[0])

    return 1.0 / max_weight < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements_noise, process_noise, particles,resampling_threshold, resampler, beta, bounds):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise, bounds)

    # Compute current particle's weight
    d_mnt, p = compute_likelihood(propagated_states, measurements_noise, beta)
    weights = particles[0] * p

    # Store
    propagated_states[0] = weights
    new_particles = propagated_states

    # Update particles
    validate_state(new_particles, bounds, d_mnt)
    particles = normalize_weights(new_particles)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles)

def get_average_state(particles):

    # Compute weighted average
    avg_x = np.sum(particles[0]*particles[1][0]) / np.sum(particles[0])
    avg_y = np.sum(particles[0]*particles[1][1]) / np.sum(particles[0])

    return [avg_x, avg_y]

def test_diverge(ERR, err_max=200):
    if ERR[-1] > err_max: #Si l'erreur est de plus de 500m il y a un probleme
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Divergence of the algorithm")
        print(f"dt = {dt}")
        print(f"process_noise = {process_noise}")
        print(f"measurements_noise = {measurements_noise}")
        print(f"V = {v_x, v_y, v_z}")
        print(f"pos_std = {lat_std, lon_std}")
        print(f"speed_std = {v_x_std, v_y_std, v_z_std}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return True #Alors on arrete
    return(False)

def set_dt(ti, pti = T[0,]):
    t = ti.split(":")
    t[2] = t[2].split(".")
    t = float(t[0])*60*60+float(t[1])*60+float(t[2][0]) + float(t[2][1])/1000.

    pt = pti.split(":")
    pt[2] = pt[2].split(".")
    pt = float(pt[0])*60*60+float(pt[1])*60+float(pt[2][0]) + float(pt[2][1])/1000.

    return(t - pt, t)

if __name__ == '__main__':
    print("~~~Start of the algorithm~~~")

    x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0,:]), np.min(coord2cart((LAT, LON))[1,:])
    x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0,:]), np.max(coord2cart((LAT, LON))[1,:])
    bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
    particles = initialize_particles_uniform(n_particles, bounds)

    #For the update
    resampler = Resampler()
    resampling_threshold = 0.5*n_particles

    dt, t = set_dt(T[steps,], T[0,])

    t_i = 0 #1*int(1/3*T.shape[0])
    t_f = T.shape[0]

    v_x = V_X[0,]
    v_y = V_Y[0,]
    v_z = V_Z[0,]
    lat = LAT[0,]
    lon = LON[0,]
    lat_std = LAT_STD[0,]
    lon_std = LON_STD[0,]
    v_x_std = V_X_STD[0,]
    v_y_std = V_Y_STD[0,]
    v_z_std = V_Z_STD[0,]

    if bool_display:
        """ Création des isobates """
        plt.ion()


        x = np.linspace(-np.min(LON), 120, 100)
        y = np.linspace(-120, 120, 100)
        X, Y = np.meshgrid(x, y)

        print("Processing..")
        r = range(t_i,t_f,steps)

    else : r = tqdm(range(t_i,t_f,steps))

    fig, ax = plt.subplots()
    TIME = []; BAR = []; SPEED = []; ERR = []
    for i in r:

        """Set data"""
        _, t = set_dt(T[i,]) #même dt pour tout t
        v_x = V_X[i,]
        v_y = V_Y[i,]
        v_z = V_Z[i,]
        lat = LAT[i,]
        lon = LON[i,]
        x_gps, y_gps = coord2cart((lat,lon)).flatten()
        lat_std = LAT_STD[i,]
        lon_std = LON_STD[i,]
        v_x_std = V_X_STD[i,]
        v_y_std = V_Y_STD[i,]
        v_z_std = V_Z_STD[i,]

        """Processing the motion of the robot """
        # robot_forward_motion =  dt*np.sqrt(v_x**2 + v_y**2)# + v_z**2)
        robot_forward_motion =  dt*np.sqrt(v_x**2 + v_y**2)# + v_z**2)
        robot_angular_motion = np.arctan2(v_x,v_y) #Je sais pas pourquoi c'est à l'envers

        """ Processing error on measures"""
        meas_model_distance_std = 1 #50*steps*(np.sqrt(lat_std**2 + lon_std**2)) # On estime que l'erreur en z est le même que celui en lat, lon, ce qui est faux
        measurements_noise = [meas_model_distance_std] ### Attention, std est en mètres !

        """ Processing error on algorithm"""
        motion_model_forward_std = steps*np.sqrt(v_y_std**2 + v_x_std**2)# + v_z_std**2)
        motion_model_turn_std = steps*np.abs(np.arctan2((v_y + v_y_std),(v_x)) - np.arctan2((v_y),(v_x+v_x_std)))
        process_noise = [motion_model_forward_std, motion_model_turn_std]

        """Process the update"""
        t0 = time.time()
        particles = update(robot_forward_motion, robot_angular_motion,\
                           measurements_noise, process_noise, particles,\
                            resampling_threshold, resampler, beta = 0.1, bounds = bounds)

        """ Affichage en temps réel """
        if bool_display:
            ax.cla()
            print("Temps de calcul: ",time.time() - t0)
            t1 = time.time()
            ax.plot(coord2cart((LAT[t_i:t_f,], LON[t_i:t_f,]))[0,:], coord2cart((LAT[t_i:t_f,], LON[t_i:t_f,]))[1,:])
            ax.set_title("Particle filter with {} particles".format(n_particles))
            ax.set_xlim([x_gps_min - 100,x_gps_max + 100])
            ax.set_ylim([y_gps_min - 100,y_gps_max + 100])
            ax.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée', s = 100)
            ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles") # Affiche toutes les particules
            bx, by = get_average_state(particles)[0], get_average_state(particles)[1] #barycentre des particules
            ax.scatter(bx, by , color = 'green', label = 'Estimation of particles')

            ax.legend()
            plt.pause(0.00001)
            print("Temps d'affichage: ",time.time()-t1,"\n")

        TIME.append(t)
        ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))
        BAR.append([get_average_state(particles)[0],get_average_state(particles)[1]])
        SPEED.append(np.sqrt(v_x**2 + v_y**2))# + v_z**2))
        # if test_diverge(ERR) : break #Permet de voir si l'algorithme diverge et pourquoi.

    DT = time.time() - T_start
    print(f"Total time: {int(DT/60/60)}h{int(DT/60)}min{int(DT-DT//60*60)}s")
    """ Affichage final """
    BAR = np.array(BAR)
    LAT, LON = LAT[t_i:t_f,], LON[t_i:t_f,]

    plt.suptitle(f"Algorithm with\n{n_particles} particles\n{steps} steps between measures with beams")
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    print("Display the error and the final result..")
    ax1.set_title("Barycentre")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.plot(coord2cart((LAT,LON))[0,:], coord2cart((LAT,LON))[1,:],label='true position')
    ax1.scatter(BAR[:,0], BAR[:,1], color='red', s = 1.2, label='barycentre of the particle')
    ax1.legend()

    ax2.set_title("Error function.")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("error (m)")
    ax2.plot(TIME, ERR, color = 'b', label = 'erreur')
    ax2.legend()

    ax3.set_title("Vitesse")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("||v|| [m/s]")
    ax3.plot(TIME, SPEED, label = 'speed')
    ax3.legend()

    print("Computing the diagrams..")

    plt.show()

print("~~~End of the algorithm~~~")
