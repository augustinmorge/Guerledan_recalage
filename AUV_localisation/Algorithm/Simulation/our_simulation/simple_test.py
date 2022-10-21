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
    # Initialize particles with uniform weight distribution
    particles = []
    weight = 1.0 / n_particles
    for i in range(n_particles):
        # Add particle i
        particles.append(
            [weight, [ \
                np.random.uniform(-5, 5, 1)[0],
                np.random.uniform(-5, 5, 1)[0]]])
    return(particles)

def get_max_weight(particles):
    return max([weighted_sample[0] for weighted_sample in particles])

def normalize_weights(weighted_samples):

    # Compute sum weighted samples
    sum_weights = 0.0
    for weighted_sample in weighted_samples:
        sum_weights += weighted_sample[0]

    # Check if weights are non-zero
    if sum_weights < 1e-15:
        print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

        # Set uniform weights
        return [[1.0 / len(weighted_samples), weighted_sample[1]] for weighted_sample in weighted_samples]

    # Return normalized weights
    return [[weighted_sample[0] / sum_weights, weighted_sample[1]] for weighted_sample in weighted_samples]

def propagate_sample(sample, forward_motion, angular_motion):
    motion_model_forward_std = 0.1
    motion_model_turn_std = 0.20
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
    propagated_sample = copy.deepcopy(sample)
    # print(propagated_sample)
    # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
    forward_displacement = np.random.normal(forward_motion, process_noise[0], 1)[0]
    angular_motion = np.random.normal(angular_motion, process_noise[1],1)[0]

    # 2. move forward
    propagated_sample[0] += forward_displacement*cos(angular_motion)
    propagated_sample[1] += forward_displacement*sin(angular_motion)

    # Make sure we stay within cyclic world
    return (propagated_sample)

def compute_likelihood(sample, measurement):

    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]


    # Initialize measurement likelihood
    likelihood_sample = 1.0

    # Map difference true and expected distance measurement to probability
    beta = 0.1
    # p_z_given_x_distance = np.exp(-beta*(np.abs(sample[0]-measurement[0]))*(np.abs(sample[1]-measurement[1]))/(2*measurement_noise[0]**2))
    p_z_given_x_distance = np.exp(-beta*(np.sqrt((sample[0]-measurement[0])**2)+(sample[1]-measurement[1])**2)/(2*measurement_noise[0]**2))

    # Incorporate likelihoods current landmark
    likelihood_sample *= p_z_given_x_distance

    # Return importance weight based on all landmarks
    return likelihood_sample

def needs_resampling(resampling_threshold):

    max_weight = 0
    for par in particles:
        max_weight = max(max_weight, par[0])

    return 1.0 / max_weight < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, particles,resampling_threshold, resampler):

    # Loop over all particles
    new_particles = []

    for par in particles:

        # Propagate the particle state according to the current particle
        propagated_state = propagate_sample(par[1], robot_forward_motion, robot_angular_motion)

        # Compute current particle's weight
        weight = par[0] * compute_likelihood(propagated_state, measurements)

        # Store
        new_particles.append([weight, propagated_state])

        plt.scatter(par[1][0], par[1][1], color = 'red')

    # Update particles
    particles = normalize_weights(new_particles)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles)


if __name__ == '__main__':
    import time
    dt = 0.05
    tf = 10

    n_particles = 1000
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
        print("Temps de calcul: ",time.time() - t0)

        plt.pause(0.00001)
        plt.clf()
        # print(x_gps,y_gps)
