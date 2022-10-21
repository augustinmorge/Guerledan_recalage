#!/usr/bin/env python3
# from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import copy
from resampler import Resampler


def GPS_point(t):
    return(cos(t),1/2*sin(2*t))

def initialize_particles_uniform(n_particles):
    weight = 1/n_particles
    particles = [weight*np.ones((n_particles,1)),[ \
                           np.random.uniform(-5, 5, n_particles).reshape(n_particles,1),
                           np.random.uniform(-5, 5, n_particles).reshape(n_particles,1)]]
    return(particles)

def get_max_weight(particles):
    return np.max(weighted_sample[0])

def normalize_weights(weighted_samples):

    # Check if weights are non-zero
    if np.sum(weighted_samples[0]) < 1e-15:
        print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

        # Set uniform weights
        return [1.0 / weighted_samples[0].shape[0]*np.ones((weighted_samples[0].shape[0],1)), weighted_sample[1]]

    # Return normalized weights
    return [weighted_samples[0] / np.sum(weighted_samples[0]), weighted_samples[1]]

def propagate_sample(samples, forward_motion, angular_motion):
    motion_model_forward_std = 0.1
    motion_model_turn_std = 0.20
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
    propagated_samples = copy.deepcopy(samples)
    # print(propagated_sample)
    # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
    forward_displacement = np.random.normal(forward_motion, process_noise[0], 1)[0]
    angular_motion = np.random.normal(angular_motion, process_noise[1],1)[0]

    # 2. move forward
    propagated_samples[1][0] += forward_displacement*cos(angular_motion)
    propagated_samples[1][1] += forward_displacement*sin(angular_motion)

    # Make sure we stay within cyclic world
    return (propagated_samples)

def compute_likelihood(samples, measurement):

    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Map difference true and expected distance measurement to probability
    beta = 0.1
    # p_z_given_x_distance = np.exp(-beta*(np.abs(sample[0]-measurement[0]))*(np.abs(sample[1]-measurement[1]))/(2*measurement_noise[0]**2))
    distance = np.sqrt(((samples[1][0]-measurement[0])**2)+(samples[1][1]-measurement[1])**2)
    p_z_given_x_distance = np.exp(-beta*distance/(2*measurement_noise[0]**2))

    # Return importance weight based on all landmarks
    print("p(z|measures): ",p_z_given_x_distance)
    return p_z_given_x_distance

def needs_resampling(resampling_threshold):

    max_weight = np.max(particles[0])
    print("max_weight: ", max_weight)

    return 1.0 / max_weight < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, particles,resampling_threshold, resampler):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion)

    # Compute current particle's weight
    weights = particles[0] * compute_likelihood(propagated_states, measurements)

    # Store
    propagated_states[0] = weights
    new_particles = propagated_states
    # print(new_particles[0])

    # Update particles
    particles = normalize_weights(new_particles)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        print("Need to resample")
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles)


if __name__ == '__main__':
    import time
    dt = 0.05
    tf = 10

    n_particles = 100
    particles = initialize_particles_uniform(n_particles)

    x_gps, y_gps = GPS_point(0)[0], GPS_point(0)[0]
    resampler = Resampler()
    for t in np.arange(np.random.uniform(0,2*np.pi,1),tf,dt):
        robot_forward_motion = np.sqrt((GPS_point(t)[0] - x_gps)**2 + (GPS_point(t)[1] - y_gps)**2)
        robot_angular_motion = np.arctan2((GPS_point(t)[1] - y_gps),(GPS_point(t)[0] - x_gps))

        x_gps = GPS_point(t)[0]
        y_gps = GPS_point(t)[1]

        measurements = [x_gps, y_gps]

        resampling_threshold = 0.5*n_particles

        #Affichage
        plt.ion()
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        plt.scatter(x_gps,y_gps,color='blue')
        t0 = time.time()
        particles = update(robot_forward_motion, robot_angular_motion, measurements, particles, resampling_threshold, resampler)
        # print(particles[0])
        print("Temps de calcul: ",time.time() - t0)
        for i in range(n_particles):
            plt.scatter(particles[1][0][i], particles[1][1][i], color = 'red')
        plt.pause(0.00001)
        plt.clf()
        # print(x_gps,y_gps)
