#!/usr/bin/env python3

import time
T_start = time.time()
# from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import copy
from resampler import Resampler
from data_import import *

wpt_ponton = (48.1989495, -3.0148023)
def coord2cart(coords,coords_ref=wpt_ponton):
    R = 6372800

    ly,lx = coords
    lym,lxm = coords_ref


    x_tilde = R * np.cos(ly*np.pi/180)*(lx-lxm)*np.pi/180
    y_tilde = R * (ly-lym)*np.pi/180

    return x_tilde,y_tilde

def distance_to_bottom(x,y):
    z = np.sqrt((x/2)**2 + (y/2)**2) + np.sin(x/2) + np.cos((x + y)/2)*np.cos(x/2)
    return(z)

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
        # print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

        # Set uniform weights
        return [1.0 / weighted_samples[0].shape[0]*np.ones((weighted_samples[0].shape[0],1)), weighted_samples[1]]

    # Return normalized weights
    return [weighted_samples[0] / np.sum(weighted_samples[0]), weighted_samples[1]]

def propagate_sample(samples, forward_motion, angular_motion, process_noise):
    # motion_model_forward_std = 0.1
    # motion_model_turn_std = 0.20
    # process_noise = [motion_model_forward_std, motion_model_turn_std]

    # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
    propagated_samples = copy.deepcopy(samples)
    # print(propagated_sample)
    # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
    n_particles = samples[0].shape[0]
    forward_displacement = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(-1,1)
    # forward_displacement_y = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(n_particles,1)
    angular_motion = np.random.normal(angular_motion, process_noise[1],n_particles).reshape(-1,1)

    # 2. move forward
    propagated_samples[1][0] += forward_displacement*np.cos(angular_motion)
    propagated_samples[1][1] += forward_displacement*np.sin(angular_motion)

    # Make sure we stay within cyclic world
    return (propagated_samples)

def compute_likelihood(samples, measurements, measurements_noise):

    # meas_model_distance_std = 0.4
    # meas_model_angle_std = 0.3
    # measurements_noise = [meas_model_distance_std, meas_model_angle_std]

    # Map difference true and expected distance measurement to probability
    beta = 0.1
    # distance = np.sqrt(((samples[1][0]-measurement[0])**2)+(samples[1][1]-measurement[1])**2)
    z_mbes_particule = distance_to_bottom(samples[1][0],samples[1][1])
    distance = np.abs(z_mbes_particule - measurements)
    p_z_given_x_distance = np.exp(-beta*distance/(2*measurements_noise[0]**2))

    # Return importance weight based on all landmarks
    return p_z_given_x_distance

def needs_resampling(resampling_threshold):

    max_weight = np.max(particles[0])

    return 1.0 / max_weight < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, measurements_noise, process_noise, particles,resampling_threshold, resampler):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise)

    # Compute current particle's weight
    weights = particles[0] * compute_likelihood(propagated_states, measurements, measurements_noise)

    # Store
    propagated_states[0] = weights
    new_particles = propagated_states

    # Update particles
    particles = normalize_weights(new_particles)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        # print("Ressempling..")
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles)

def get_average_state(particles):

    # Compute weighted average
    avg_x = np.sum(particles[0]*particles[1][0]) / np.sum(particles[0])
    avg_y = np.sum(particles[0]*particles[1][1]) / np.sum(particles[0])

    return [avg_x, avg_y]

def test_diverge(ERR, i, err_max=500):
    if ERR[-1] > err_max: #Si l'erreur est de plus de 500m il y a un probleme
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Divergence of the algorithm")
        print(f"dt = {dt[i,]}")
        print(f"process_noise = {[process_noise[0][i,0],process_noise[1][i,0]]}")
        print(f"measurements_noise = {[measurements_noise[0][i,0]]}")
        print(f"V = {V_X[i,], V_Y[i,], V_Z[i,]}")
        print(f"pos_std = {LAT_STD[i,], LON_STD[i,]}")
        print(f"speed_std = {V_X_STD[i,], V_Y_STD[i,], V_Z_STD[i,]}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return True #Alors on arrete
    return(False)

def dt(T, N, steps):
    dt = np.zeros((N,1))
    dt[0,] = 0
    time = [0]

    for i in range(1,N):
        t = T[i,].split(":")
        t[2] = t[2].split(".")
        t = float(t[0])*60*60+float(t[1])*60+float(t[2][0]) + float(t[2][1])/1000.

        pt = T[i-1,].split(":")
        pt[2] = pt[2].split(".")
        pt = float(pt[0])*60*60+float(pt[1])*60+float(pt[2][0]) + float(pt[2][1])/1000.

        dt[i,] = (t - pt)
        time.append(t)

    return(dt*steps, time)

if __name__ == '__main__':
    import PIL.Image as Image
    import osm_ui
    import sys

    # n_particles = int(input("Number of particles: "))
    # steps = int(input("number of steps between measures ? "))
    # bool_display = str(input("Display the particles ? [Y/]"))
    # bool_display = bool_display=="Y"
    n_particles = 1000
    steps = 25
    bool_display = False

    """ Reshape the variables """
    LAT = LAT.reshape(-1,1)
    LON = LON.reshape(-1,1)
    V_X = V_X.reshape(-1,1)
    V_Y = V_Y.reshape(-1,1)
    V_Z = V_Z.reshape(-1,1)
    LAT_STD = LAT_STD.reshape(-1,1)
    LON_STD = LON_STD.reshape(-1,1)
    V_X_STD = V_X_STD.reshape(-1,1)
    V_Y_STD = V_Y_STD.reshape(-1,1)
    V_Z_STD = V_Z_STD.reshape(-1,1)

    """ Initalize particules """
    x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0]), np.min(coord2cart((LAT, LON))[1])
    x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0]), np.max(coord2cart((LAT, LON))[1])
    bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
    particles = initialize_particles_uniform(n_particles, bounds)

    """ Initalize the resampling """
    resampler = Resampler()
    resampling_threshold = 0.5*n_particles

    """ Set dt """
    dt, TIME = dt(T, T.shape[0], steps)

    """ Processing error on measures"""
    x_gps, y_gps = coord2cart((LAT,LON))[0], coord2cart((LAT,LON))[1]
    measurements = distance_to_bottom(x_gps, y_gps)

    robot_forward_motion =  dt*np.sqrt(V_X**2 + V_Y**2 + V_Z**2)
    robot_angular_motion = np.arctan2(V_X,V_Y)
    meas_model_distance_std = steps*np.sqrt(LAT_STD**2 + LON_STD**2)
    measurements_noise = [meas_model_distance_std]

    """ Processing error on algorithm"""
    motion_model_forward_std = steps*np.sqrt(V_X_STD**2 + V_Y_STD**2 + V_Z_STD**2)
    motion_model_turn_std = steps*np.abs(np.arctan2((V_Y + V_Y_STD),(V_X)) - np.arctan2((V_Y),(V_X+V_X_STD)))
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    plt.ion()
    BAR = []; ERR = []

    print("Processing..")
    from tqdm import tqdm
    # for i in (range(0,LON.shape[0],steps)):
    for i in tqdm(range(0,LON.shape[0],steps)):

        """Process the update"""
        # print(measurements[i,0]) #, measurements_noise, process_noise, particles, resampling_threshold, resampler)
        particles = update(robot_forward_motion[i,0], robot_angular_motion[i,0], measurements[i,0], \
                            [measurements_noise[0][i,0]], [process_noise[0][i,0],process_noise[1][i,0]],\
                             particles, resampling_threshold, resampler)

        """ Affichage en temps réel """
        if bool_display:
            print("Temps de calcul: ",time.time() - t0)
            t1 = time.time()
            plt.plot(coord2cart((LAT,LON))[0,:], coord2cart((LAT,LON))[1,:])
            plt.title("Particle filter with {} particles with z = {}m".format(n_particles, measurements))
            plt.xlim([x_gps_min - 100,x_gps_max + 100])
            plt.ylim([y_gps_min - 100,y_gps_max + 100])
            plt.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée')
            # for i in range(n_particles):
            #     plt.scatter(particles[1][0][i], particles[1][1][i], color = 'red')
            plt.scatter(get_average_state(particles)[0],get_average_state(particles)[1], color = 'red', label = 'Approximation of particles')
            plt.legend()
            plt.pause(0.00001)
            plt.clf()
            print("Temps d'affichage: ",time.time()-t1,"\n")

        ERR.append(np.sqrt((x_gps[i] - get_average_state(particles)[0])**2 + (y_gps[i] - get_average_state(particles)[1])**2))
        BAR.append([get_average_state(particles)[0],get_average_state(particles)[1]])

        if test_diverge(ERR, i) : break #Permet de voir si l'algorithme diverge et pourquoi ?

    print(f"Temps de calcul total = {(time.time() - T_start)}s")
    """ Affichage final """
    plt.close()
    fig,ax = plt.subplots(1,2)
    print("Display the error and the final result..")
    ax[0].set_title(f"Error function with\n{n_particles} particles\n{steps} steps between measures.")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("error (m)")

    ax[1].set_title("Barycentre")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("y [m]")
    ax[1].plot(x_gps, y_gps,label='true position')

    for i in range(len(ERR)):
        if i == 0:
            ax[1].scatter(BAR[i][0], BAR[i][1], color='red', label='barycentre of the particle')
        if (i+1) % steps == 0:
            ax[0].scatter(TIME[i], ERR[i], color = 'b')
            ax[1].scatter(BAR[i][0], BAR[i][1], color='red', s = 1.2)
    plt.legend()
    plt.show()
    print("End the program.")
    plt.pause(100000)
