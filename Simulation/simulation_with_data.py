#!/usr/bin/env python3
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

    return np.array([x_tilde,y_tilde])

def distance_to_bottom(x,y):
    z = np.sqrt((x/2)**2 + (y/2)**2) + np.sin(x/2) + np.cos((x + y)/2)*np.cos(x/2)
    return(z)

def initialize_particles_uniform(n_particles, bounds):
    weight = 1/n_particles
    particles = [weight*np.ones((n_particles,1)),[ \
                           np.random.uniform(bounds[0][0], bounds[0][1], n_particles).reshape(n_particles,1),
                           np.random.uniform(bounds[1][0], bounds[1][1], n_particles).reshape(n_particles,1)]]
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
    forward_displacement = np.random.normal(forward_motion, process_noise[0], n_particles).reshape(n_particles,1)
    angular_motion = np.random.normal(angular_motion, process_noise[1],n_particles).reshape(n_particles,1)

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

if __name__ == '__main__':
    import time
    import PIL.Image as Image
    import osm_ui

    n_particles = 1000 #int(input("Number of particles: "))
    n_sec = 1000# int(input("Number of seconds: "))
    x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0,:]), np.min(coord2cart((LAT, LON))[1,:])
    x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0,:]), np.max(coord2cart((LAT, LON))[1,:])
    bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
    particles = initialize_particles_uniform(n_particles, bounds)

    #For the update
    resampler = Resampler()
    resampling_threshold = 0.5*n_particles

    t = T[0,]
    v_x = V_X[0,]
    v_y = V_Y[0,]
    lat = LAT[0,]
    lon = LON[0,]
    lat_std = LAT_STD[0,]
    lon_std = LON_STD[0,]
    v_x_std = V_X_STD[0,]
    v_y_std = V_Y_STD[0,]

    # lac = Image.open("./imgs/ortho_sat_2016_guerledan.tif")
    # # axes = osm_ui.plot_map(lac, (-3.118111, -2.954274), (48.183105, 48.237852), "Mis à l'eau de l'AUV")
    # lac_coords_min = coord2cart((3.118111, 48.183105)).flatten()
    # lac_coords_max = coord2cart((-2.954274, 48.237852)).flatten()
    # axes = osm_ui.plot_map(lac, (lac_coords_min[0], lac_coords_max[0]), (lac_coords_min[1], lac_coords_max[1]), "Mis à l'eau de l'AUV")
    # osm_ui.plot_xy_add(axes, LON, LAT)
    # axes.legend(("ins",))

    plt.ion()

    TIME = []; ERR = []

    print("Start to display the log..")
    from tqdm import tqdm
    steps = 10
    for i in tqdm(range(0,LON.shape[0],steps)):
        #Coordinates in cartesian
        px_gps, py_gps = coord2cart((lat,lon)).flatten()

        #Update data
        t = T[i,]
        v_x = V_X[i,]
        v_y = V_Y[i,]
        lat = LAT[i,]
        lon = LON[i,]
        x_gps, y_gps = coord2cart((lat,lon)).flatten()
        lat_std = LAT_STD[i,]
        lon_std = LON_STD[i,]
        v_x_std = V_X_STD[i,]
        v_y_std = V_Y_STD[i,]

        robot_forward_motion = np.sqrt((x_gps - px_gps)**2 + (y_gps - py_gps)**2)
        robot_angular_motion = np.arctan2((y_gps - py_gps),(x_gps - px_gps))


        measurements = distance_to_bottom(x_gps, y_gps)

        meas_model_distance_std = steps*np.sqrt(lat_std**2 + lon_std**2)
        # meas_model_angle_std = 0.3
        measurements_noise = [meas_model_distance_std] ### Attention, std est en mètres !

        motion_model_forward_std = steps*np.sqrt(v_x_std**2+v_y_std**2)
        # v_px_std = V_X_STD[max(0,steps*(i-1)),]
        # v_py_std = V_Y_STD[max(0,steps*(i-1)),]
        # motion_model_turn_std = np.abs(np.arctan2((v_py_std-v_y_std),(v_px_std-v_x_std)))
        motion_model_turn_std = 0.2
        process_noise = [motion_model_forward_std, motion_model_turn_std]
        # print(f"process_noise={process_noise}")
        t0 = time.time()
        particles = update(robot_forward_motion, robot_angular_motion, measurements, measurements_noise, process_noise, particles, resampling_threshold, resampler)
        # print("Temps de calcul: ",time.time() - t0)

        # #Affichage
        #
        # t1 = time.time()
        # plt.title("Particle filter with {} particles".format(n_particles))
        # plt.xlim([x_gps_min,x_gps_max])
        # plt.ylim([y_gps_min,y_gps_max])
        # plt.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée')
        # # for i in range(n_particles):
        # #     plt.scatter(particles[1][0][i], particles[1][1][i], color = 'red')
        # plt.scatter(get_average_state(particles)[0],get_average_state(particles)[1], color = 'red', label = 'Approximation of particles')
        # plt.legend()
        # plt.pause(0.00001)
        # plt.clf()
        # print("Temps d'affichage: ",time.time()-t1,"\n")

        TIME.append(t)
        ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))

    plt.figure()
    plt.title(f"Error function with {n_particles} particles")
    plt.plot(TIME, ERR)
    plt.xlabel("time [s]")
    plt.ylabel("error (m)")
    plt.show()
    plt.pause(100)
