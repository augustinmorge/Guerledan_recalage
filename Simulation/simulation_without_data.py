#!/usr/bin/env python3
# from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from simulation_helper.resampler import Resampler

def distance_to_bottom(x,y):
    z = np.sqrt((x/2)**2 + (y/2)**2) + np.sin(x/2) + np.cos((x + y)/2)*np.cos(x/2)
    return(z)

def GPS_point(t, alpha = 0.25):
    return(120.*cos(0.25*alpha*t),
           120.*sin(0.5*alpha*t))


def initialize_particles_uniform(n_particles):
    weight = 1/n_particles
    particles = [weight*np.ones((n_particles,1)),[ \
                           np.random.uniform(-120, 120, n_particles).reshape(n_particles,1),
                           np.random.uniform(-120, 120, n_particles).reshape(n_particles,1)]]
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

def propagate_sample(samples, forward_motion, angular_motion):
    motion_model_forward_std = 0.1
    motion_model_turn_std = 0.20
    process_noise = [motion_model_forward_std, motion_model_turn_std]

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

def compute_likelihood(samples, measurement):

    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Map difference true and expected distance measurement to probability
    beta = 0.1
    # distance = np.sqrt(((samples[1][0]-measurement[0])**2)+(samples[1][1]-measurement[1])**2)
    z_mbes = distance_to_bottom(samples[1][0],samples[1][1])
    distance = np.abs(z_mbes - measurement)
    p_z_given_x_distance = np.exp(-beta*distance/(2*measurement_noise[0]**2))

    # Return importance weight based on all landmarks
    return p_z_given_x_distance

def needs_resampling(resampling_threshold):

    max_weight = np.max(particles[0])

    return 1.0 / max_weight < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, particles,resampling_threshold, resampler):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion)

    # Compute current particle's weight
    weights = particles[0] * compute_likelihood(propagated_states, measurements)

    # Store
    propagated_states[0] = weights
    new_particles = propagated_states

    # Update particles
    particles = normalize_weights(new_particles)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        print("Ressempling..")
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles)

def get_average_state(particles):

    # Compute weighted average
    avg_x = np.sum(particles[0]*particles[1][0]) / np.sum(particles[0])
    avg_y = np.sum(particles[0]*particles[1][1]) / np.sum(particles[0])

    return [avg_x, avg_y]

if __name__ == '__main__':
    import time

    n_particles = 1000#int(input("Number of particles: "))
    n_sec = 1000#int(input("Number of seconds: "))
    particles = initialize_particles_uniform(n_particles)

    x_gps, y_gps = GPS_point(0)[0], GPS_point(0)[1]
    resampler = Resampler()

    plt.ion()
    t_ini = time.time()
    ERR = []; TIME = []

    x = np.linspace(-120, 120, 100)
    y = np.linspace(-120, 120, 100)
    X, Y = np.meshgrid(x, y)

    # Calcul des valeurs de z
    Z = distance_to_bottom(X, Y)
    mult = 2
    extent = (-mult*120, mult*120, -mult*120, mult*120)
    N = 10
    min_z = np.min(Z)
    max_z = np.max(Z)
    print(min_z,max_z)
    d = lambda k : min_z + k*(max_z - min_z)*1/N
    levels = [d(k) for k in range(N)]
    fig, ax = plt.subplots()

    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
    cmap=cm.gray, extent = extent)
    CS = ax.contour(Z, levels, origin='lower', cmap='flag', extend='both',
    linewidths=2, extent = extent)
    # Thicken the zero contour.
    CS.collections[6].set_linewidth(4)
    ax.clabel(CS, levels[1::2],  # label every second level
    inline=True, fmt='%1.1f', fontsize=14)
    # make a colorbar for the contour lines
    CB = fig.colorbar(CS, shrink=0.8)
    ax.set_title('Lines with colorbar')
    # We can still add a colorbar for the image, too.
    CBI = fig.colorbar(im, orientation='horizontal', shrink=0.8)
    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.
    l, b, w, h = ax.get_position().bounds
    ll, bb, ww, hh = CB.ax.get_position().bounds


    while(time.time() - t_ini < n_sec):
        t = time.time() - t_ini
        robot_forward_motion = np.sqrt((GPS_point(t)[0] - x_gps)**2 + (GPS_point(t)[1] - y_gps)**2)
        robot_angular_motion = np.arctan2((GPS_point(t)[1] - y_gps),(GPS_point(t)[0] - x_gps))

        x_gps = GPS_point(t)[0]
        y_gps = GPS_point(t)[1]

        measurements = distance_to_bottom(x_gps, y_gps)

        resampling_threshold = 0.5*n_particles

        t0 = time.time()
        particles = update(robot_forward_motion, robot_angular_motion, measurements, particles, resampling_threshold, resampler)
        print("Temps de calcul: ",time.time() - t0)

        #Affichage
        t1 = time.time()
        ax.set_title("Particle filter with {} particles".format(n_particles))
        # ax.set_xlim([-1200,120])
        # ax.set_ylim([-1200,120])
        ax.set_xlim([x_gps - 120,x_gps + 120])
        ax.set_ylim([y_gps - 120,y_gps + 120])
        ax.scatter(x_gps,y_gps,color='blue', label = 'True position')
        # for i in range(n_particles):
        #     ax.scatter(particles[1][0][i], particles[1][1][i], color = 'red')
        ax.scatter(get_average_state(particles)[0],get_average_state(particles)[1], color = 'red', label = 'Approximation of particles')
        ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles",alpha=particles[0][:,0]/pow(np.max(particles[0][:,0]),2/3)) # Affiche toutes les particules

        # Tracé des isobates
        # plt.contour(X, Y, Z, levels)

        im = ax.imshow(Z, interpolation='bilinear', origin='lower',
        cmap=cm.gray, extent = extent)

        CS = ax.contour(Z, levels, origin='lower', cmap='flag', extend='both',
        linewidths=2, extent = extent)

        CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

        ax.legend()
        plt.pause(0.00001)
        ax.cla()
        print("Temps d'affichage: ",time.time()-t1,"\n")
        TIME.append(t)
        ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))

    plt.plot(TIME, ERR)
    plt.title(f"Error function with {n_particles} particles")
    plt.xlabel("time [s]")
    plt.ylabel("error (m)")
    plt.show()
    plt.pause(100)
