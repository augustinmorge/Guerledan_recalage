#!/usr/bin/env python3

from storage_afternoon.data_import import *
n_particles = int(input("Number of particles: "))
steps = int(input("number of steps between measures ? "))
bool_display = (str(input("Display the particles ? [Y/]"))=="Y")

import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
from resampler import Resampler
from tqdm import tqdm
import sys
file_path = os.path.dirname(os.path.abspath(__file__))

def sawtooth(x):
    return(2*np.arctan(np.tan(x/2)))

def distance_to_bottom(xy,mnt):
    d_mnt, indices = kd_tree.query(xy)  #Utilise KDTree pour calculer les distances
    Z = mnt[indices,2] # Récupère les altitudes des points les plus proches
    return d_mnt, Z

def initialize_particles_uniform(n_particles, bounds):
    particles = [1/n_particles*np.ones((n_particles,1)),[ \
                           np.random.uniform(bounds[0][0]-20, bounds[0][1]+20, n_particles).reshape(-1,1),
                           np.random.uniform(bounds[1][0]-20, bounds[1][1]+20, n_particles).reshape(-1,1)]]
    return(particles)

def normalize_weights(weighted_samples):
    return [weighted_samples[0] / np.sum(weighted_samples[0]), weighted_samples[1]]

def validate_state(state, bounds, d_mnt):
    # x_min, x_max = bounds[0][0] - 10., bounds[0][1] + 10.
    # y_min, y_max = bounds[1][0] - 10., bounds[1][1] + 10.
    #
    # weights = state[0]
    # coords  = state[1]
    # weights[(coords[0] < x_min) | (coords[0] > x_max) | (coords[1] < y_min) | (coords[1] > y_max)] = 0
    # weights[d_mnt > 1] = 0 # If we are out of the MNT
    # if np.sum(weights) == 0: sys.exit()
    return(state)

def propagate_sample(samples, forward_motion, angular_motion, process_noise, bounds):

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

def compute_likelihood(samples, measurements, measurements_noise, beta, previous_z_mbes_particule):
    d_mnt, z_mbes_particule = distance_to_bottom(np.hstack((samples[1][0],samples[1][1])),MNT)
    d_mbes_particule = z_mbes_particule - previous_z_mbes_particule

    # Map difference true and expected distance measurement to probability
    distance = np.abs(d_mbes_particule - measurements)
    # p_z_given_x_distance = np.exp(-beta*distance/(measurements_noise[0]**2))
    p_z_given_x_distance = np.exp(-beta*distance)

    # Return importance weight based on all landmarks
    previous_z_mbes_particule = z_mbes_particule
    return d_mnt, p_z_given_x_distance, previous_z_mbes_particule

def needs_resampling(resampling_threshold):
    return 1.0 / np.max(particles[0]) < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, \
            measurements_noise, process_noise, particles,resampling_threshold,\
            resampler, beta, bounds, previous_z_mbes_particule):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise, bounds)

    # Compute current particle's weight
    d_mnt, p, previous_z_mbes_particule = compute_likelihood(propagated_states, measurements, measurements_noise, beta, previous_z_mbes_particule)

    particules = validate_state(propagated_states, bounds, d_mnt)

    # Update the probability of the particle
    propagated_states[0] = particles[0] * p

    # Update particles
    particles = normalize_weights(propagated_states)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL

    return(particles, previous_z_mbes_particule)

def get_average_state(particles):

    # Compute weighted average
    avg_x = np.sum(particles[0]*particles[1][0]) / np.sum(particles[0])
    avg_y = np.sum(particles[0]*particles[1][1]) / np.sum(particles[0])

    return [avg_x, avg_y]

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
    bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
    particles = initialize_particles_uniform(n_particles, bounds)

    _, previous_z_mbes_particule = distance_to_bottom(np.hstack((particles[1][0],particles[1][1])),MNT)

    #For the update
    resampler = Resampler()
    resampling_threshold = 0.5*n_particles

    idx_ti = 0 + steps #int(1/3*T.shape[0]) #
    idx_tf =  dvl_T.shape[0] #int(4/5*T.shape[0]) #

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

    x_gps, y_gps = coord2cart((lat,lon)).flatten()

    if bool_display:
        """ Création des isobates """
        plt.ion()


        x = np.linspace(-np.min(LON), 120, 100)
        y = np.linspace(-120, 120, 100)
        X, Y = np.meshgrid(x, y)

        # from PIL import Image
        # image = Image.open("./storage_afternoon/MNT_G1.png")
        # image.show()

        print("Processing..")
        r = range(idx_ti,idx_tf,steps)


    else : r = tqdm(range(idx_ti,idx_tf,steps))
    fig, ax = plt.subplots()
    TIME = []; BAR = []; SPEED = []; ERR = []
    STD_X = []; STD_Y = []
    MEASUREMENTS = []
    beta = 1/5.
    previous_MBES = (dvl_BM1R[0,] + dvl_BM2R[0,] + dvl_BM3R[0,] + dvl_BM4R[0,])/4 #MBES_Z[0,]


    x_gps, y_gps = coord2cart((LAT[0,],LON[0,])).flatten()
    _, previous_measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)


    for i in r:

        """Set data"""
        t = dvl_T[i,]
        yaw = sawtooth(-YAW[i,]+np.pi/2)
        v_x = (dvl_VE[i,]*np.cos(-np.pi/4) + np.sin(-np.pi/4)*dvl_VN[i,])*np.cos(yaw)
        v_y = (dvl_VE[i,]*np.sin(-np.pi/4) + np.cos(-np.pi/4)*dvl_VN[i,])*np.sin(yaw)
        # v_x = dvl_VE[i,]*np.cos(YAW[i,] - np.pi/2)
        # v_y = dvl_VN[i,]*np.sin(YAW[i,] - np.pi/2)

        # v_z = dvl_VZ[i,]
        v_std = 0.02 #dvl_VSTD[i,]

        # measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)[1] - previous_measurements
        # _, previous_measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)

        mean_range_dvl = (dvl_BM1R[i,] + dvl_BM2R[i,] + dvl_BM3R[i,] + dvl_BM4R[i,])/4
        measurements = mean_range_dvl - previous_MBES #MBES_Z[i,] - previous_MBES
        previous_MBES = mean_range_dvl #MBES_Z[i,]

        """Processing the motion of the robot """
        robot_forward_motion =  dt*np.sqrt(v_x**2 + v_y**2)# + v_z**2)
        # robot_angular_motion = sawtooth(-yaw+np.pi/2) #np.arctan2(v_y,v_x) #Je sais pas pourquoi c'est à l'envers
        robot_angular_motion = yaw #np.arctan2(v_y,v_x) #Je sais pas pourquoi c'est à l'envers

        """ Processing error on measures"""
        meas_model_distance_std = None
        measurements_noise = [meas_model_distance_std] ### Attention, std est en mètres !

        """ Processing error on algorithm"""
        # motion_model_forward_std = steps*np.abs(v_std)
        motion_model_forward_std = steps*np.abs(v_std)

        # motion_model_turn_std = np.abs(sawtooth(np.arctan2((v_x + np.sign(v_x)*v_std),(v_y)) - np.arctan2((v_x),(v_y+np.sign(v_y)*v_std))))
        motion_model_turn_std = YAW_STD[i,]
        process_noise = [motion_model_forward_std, motion_model_turn_std]

        """Process the update"""
        t0 = time.time()
        particles, previous_z_mbes_particule = update(robot_forward_motion, robot_angular_motion, measurements,\
                                               measurements_noise, process_noise, particles,\
                                                resampling_threshold, resampler, beta, bounds,\
                                                previous_z_mbes_particule)


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

        _, measurements_mnt = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
        lat = LAT[i,]
        lon = LON[i,]
        x_gps, y_gps = coord2cart((lat,lon)).flatten()
        ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))
        BAR.append([get_average_state(particles)[0],get_average_state(particles)[1]])
        SPEED.append(np.sqrt(v_x**2 + v_y**2))# + v_z**2))

        var = np.std(np.column_stack((particles[1][0],particles[1][1])),axis=0)
        STD_X.append(var[0])
        STD_Y.append(var[1])

        _, new_measurements_mnt = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
        d_measurements_mnt = new_measurements_mnt - measurements_mnt
        MEASUREMENTS.append([d_measurements_mnt, measurements])

        #Test if the algorithm diverge and why
        if test_diverge(ERR, 1000) : break



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
    MEASUREMENTS = np.array(MEASUREMENTS)

    plt.suptitle(f"Algorithm with\n{n_particles} particles; 1/{steps} data log used\nTotal time:{int(elapsed_time)}s")
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3 = plt.subplot2grid((3, 2), (2, 1))

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
    ax2.legend()

    ax3.set_title("Difference of measurements = {}.".format(np.abs(np.mean(MEASUREMENTS[:,0]) - np.mean(MEASUREMENTS[:,1]))))
    ax3.set_xlabel("time [min]")
    ax3.set_ylabel("error (m)")
    ax3.scatter(TIME, MEASUREMENTS[:,0], color = 'b', label = 'measurements from the MNT')
    ax3.scatter(TIME, MEASUREMENTS[:,1], color = 'r', label = 'measurements from the MBES')
    ax3.legend()

    ax4.set_title("Vitesse")
    ax4.set_xlabel("time [min]")
    ax4.set_ylabel("||v|| [m/s]")
    ax4.plot(TIME, SPEED, label = 'speed')
    ax4.legend()

    print("Computing the diagrams..")

    plt.show()
    if bool_display:plt.pause(100)


print("~~~End of the algorithm~~~")
