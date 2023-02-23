#!/usr/bin/env python3

n_particles = int(input("Number of particles: "))
steps = int(input("number of steps between measures ? "))
bool_display = (str(input("Display the particles ? [Y/]"))=="Y")

import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
from simulation_helper.resampler import Resampler
from storage.data_import import *
import sys
from tqdm import tqdm
file_path = os.path.dirname(os.path.abspath(__file__))

def sawtooth(x):
    return(2*np.arctan(np.tan(x/2)))

def distance_to_bottom(xy,mnt):
    d_mnt, indices = kd_tree.query(xy)  #Utilise KDTree pour calculer les distances
    Z = mnt[indices,2] # Récupère les altitudes des points les plus proches
    return d_mnt, Z

def initialize_particles_uniform(n_particles, bounds):
    particles = [1/n_particles*np.ones((n_particles,1)),[ \
                           np.random.uniform(bounds[0][0], bounds[0][1], n_particles).reshape(-1,1),
                           np.random.uniform(bounds[1][0], bounds[1][1], n_particles).reshape(-1,1)]]
    return(particles)

def normalize_weights(weighted_samples):
    return [weighted_samples[0] / np.sum(weighted_samples[0]), weighted_samples[1]]

def validate_state(state, bounds, d_mnt):
    # # x_min, x_max = bounds[0][0] - 10., bounds[0][1] + 10.
    # # y_min, y_max = bounds[1][0] - 10., bounds[1][1] + 10.
    # #
    # weights = state[0]
    # # coords  = state[1]
    # # weights[(coords[0] < x_min) | (coords[0] > x_max) | (coords[1] < y_min) | (coords[1] > y_max)] = 0
    # weights[d_mnt > 1] = 0 # If we are out of the MNT
    # # if np.sum(weights) == 0: sys.exit()
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

def compute_likelihood(samples, measurements, measurements_noise, beta):
    d_mnt, z_mbes_particule = distance_to_bottom(np.hstack((samples[1][0],samples[1][1])),MNT)

    # Map difference true and expected distance measurement to probability
    distance = np.abs(z_mbes_particule - measurements)
    # p_z_given_x_distance = np.exp(-beta*distance/(measurements_noise[0]**2))
    p_z_given_x_distance = np.exp(-beta*distance)

    # Return importance weight based on all landmarks
    return d_mnt, p_z_given_x_distance

def needs_resampling(resampling_threshold):
    return 1.0 / np.max(particles[0]) < resampling_threshold

ct_resampling = 0
def update(roboidx_tforward_motion, robot_angular_motion, measurements, measurements_noise, process_noise, particles,resampling_threshold, resampler, beta, bounds):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, roboidx_tforward_motion, robot_angular_motion, process_noise, bounds)

    # Compute current particle's weight
    d_mnt, p = compute_likelihood(propagated_states, measurements, measurements_noise, beta)

    particules = validate_state(propagated_states, bounds, d_mnt)

    # Update the probability of the particle
    propagated_states[0] = particles[0] * p

    # Update particles
    particles = normalize_weights(propagated_states)

    # Resample if needed
    if needs_resampling(resampling_threshold):
        particles = resampler.resample(particles, n_particles) #1 = MULTINOMIAL
        global ct_resampling
        ct_resampling += 1
    return(particles)

def get_average_state(particles):

    # Compute weighted average
    avg_x = np.sum(particles[0]*particles[1][0]) / np.sum(particles[0])
    avg_y = np.sum(particles[0]*particles[1][1]) / np.sum(particles[0])

    return [avg_x, avg_y]

def get_std_state(particles):

    # Compute weighted average
    std_x = np.sqrt(np.sum(particles[0]*particles[1][0]**2)/n_particles - (np.sum(particles[0]*particles[1][0])/n_particles)**2) # / np.sum(particles[0])
    std_y = np.sqrt(np.sum(particles[0]*particles[1][1]**2)/n_particles - (np.sum(particles[0]*particles[1][1])/n_particles)**2) # / np.sum(particles[0])

    return std_x, std_y
def test_diverge(ERR, err_max=500):
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

# def set_dt(ti, pti = T[0,]):
#     t = ti.split(":")
#     t[2] = t[2].split(".")
#     t = float(t[0])*60*60+float(t[1])*60+float(t[2][0]) + float(t[2][1])/1000.
#
#     pt = pti.split(":")
#     pt[2] = pt[2].split(".")
#     pt = float(pt[0])*60*60+float(pt[1])*60+float(pt[2][0]) + float(pt[2][1])/1000.
#
#     return(t - pt, t)

if __name__ == '__main__':
    print("~~~Start of the algorithm~~~")

    x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0,:]), np.min(coord2cart((LAT, LON))[1,:])
    x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0,:]), np.max(coord2cart((LAT, LON))[1,:])
    bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
    particles = initialize_particles_uniform(n_particles, bounds)

    #For the update
    resampler = Resampler()
    resampling_threshold = 0.5*n_particles

    idx_ti = 0 #int(1/3*T.shape[0])
    idx_tf = T.shape[0] #int(4/5*T.shape[0]) #

    dt = T[steps,] - T[0,]
    tini = T[idx_ti,]

    if bool_display:
        """ Création des isobates """
        plt.ion()


        x = np.linspace(-np.min(LON), 120, 100)
        y = np.linspace(-120, 120, 100)
        X, Y = np.meshgrid(x, y)

        # from PIL import Image
        # image = Image.open("./storage/MNT_G1.png")
        # image.show()

        print("Processing..")
        r = range(idx_ti,idx_tf,steps)


    else : r = tqdm(range(idx_ti,idx_tf,steps))
    fig, ax = plt.subplots()
    TIME = []; BAR = []; SPEED = []; ERR = []
    STD_X = []; STD_Y = []
    XGPS, YGPS = coord2cart((LAT, LON))

    # beta = 1/100.
    beta = 10**(-1.37)
    for i in r:

        """Set data"""
        t = T[i,]
        v_x = V_X[i,]
        v_y = V_Y[i,]
        v_z = V_Z[i,]
        lat = LAT[i,]
        lon = LON[i,]
        x_gps, y_gps = coord2cart((lat,lon)).flatten()
        _, measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
        lat_std = LAT_STD[i,]
        lon_std = LON_STD[i,]
        v_x_std = V_X_STD[i,]
        v_y_std = V_Y_STD[i,]
        v_z_std = V_Z_STD[i,]

        """Processing the motion of the robot """
        roboidx_tforward_motion =  dt*np.sqrt(v_x**2 + v_y**2)# + v_z**2)
        robot_angular_motion = np.arctan2(v_y,v_x) #Je sais pas pourquoi c'est à l'envers

        """ Processing error on measures"""
        meas_model_distance_std = None #1 #50*steps*(np.sqrt(lat_std**2 + lon_std**2)) # On estime que l'erreur en z est le même que celui en lat, lon, ce qui est faux
        measurements_noise = [meas_model_distance_std] ### Attention, std est en mètres !

        """ Processing error on algorithm"""
        motion_model_forward_std = steps*np.sqrt(v_y_std**2 + v_x_std**2)# + v_z_std**2)
        motion_model_turn_std = np.abs(sawtooth(np.arctan2((v_x + np.sign(v_x)*v_x_std),(v_y)) - np.arctan2((v_x),(v_y+np.sign(v_y)*v_y_std))))
        process_noise = [motion_model_forward_std, motion_model_turn_std]

        """Process the update"""
        t0 = time.time()
        particles = update(roboidx_tforward_motion, robot_angular_motion, measurements,\
                           measurements_noise, process_noise, particles,\
                            resampling_threshold, resampler, beta = beta, bounds = bounds)

        """ Affichage en temps réel """
        if bool_display:
            if i%1 == 0:
                # idx_ti = 0
                # idx_tf = int(T.shape[0]/3)
                lat = LAT[i,]
                lon = LON[i,]
                x_gps, y_gps = coord2cart((lat,lon)).flatten()
                ax.cla()
                print("Temps de calcul: ",time.time() - t0)
                t1 = time.time()
                ax.plot(coord2cart((LAT[idx_ti:idx_tf,], LON[idx_ti:idx_tf,]))[0,:], coord2cart((LAT[idx_ti:idx_tf,], LON[idx_ti:idx_tf,]))[1,:])
                ax.set_title(f"Particle filter using MBES with {n_particles} particles") #with z = {}m".format(n_particles, measurements))
                # ax.set_xlim([x_gps_min - 100,x_gps_max + 100])
                # ax.set_ylim([y_gps_min - 100,y_gps_max + 100])
                ax.set_xlim([np.min(XGPS[idx_ti:idx_tf,]-20),np.max(XGPS[idx_ti:idx_tf,])])
                ax.set_ylim([np.min(YGPS[idx_ti:idx_tf,]),np.max(YGPS[idx_ti:idx_tf,])])
                bx, by = get_average_state(particles)[0], get_average_state(particles)[1] #barycentre des particules
                scatter1 = ax.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée', s = 100)
                # scatter2 = ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles",alpha=particles[0][:,0]/pow(np.max(particles[0][:,0]),2/3))
                scatter2 = ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles",alpha=particles[0][:,0]/np.max(particles[0][:,0]))
                # scatter2 = ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles") #,alpha=particles[0][:,0]/pow(np.max(particles[0][:,0]),2/3))
                scatter3 = ax.scatter(bx, by , color = 'green', label = 'Estimation of particles')

                plt.legend()


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

        # std = np.std(np.column_stack((particles[1][0],particles[1][1])),axis=0)
        # STD_X.append(std[0])
        # STD_Y.append(std[1])

        std_x, std_y = get_std_state(particles)
        STD_X.append(std_x)
        STD_Y.append(std_y)
        #Test if the algorithm diverge and why
        # if test_diverge(ERR, 500) : break

        # if i == 0:
        #     plt.pause(10)


    print(f"Resampling used: {ct_resampling} ({ct_resampling/((idx_tf - idx_ti)/steps)*100}%)")
    # elapsed_time = time.perf_counter() - start_time
    # print(time.perf_counter())
    # print(start_time)
    print("Elapsed time: {:.2f} seconds".format(time.perf_counter() - start_time))

    """ Affichage final """
    TIME = (np.array(TIME) - tini)/60.
    BAR = np.array(BAR)
    LAT, LON = LAT[idx_ti:idx_tf,], LON[idx_ti:idx_tf,]
    STD_X = np.array(STD_X).squeeze()
    STD_Y = np.array(STD_Y).squeeze()
    NORM_STD = np.sqrt(STD_X**2 + STD_Y**2)
    max_std = 3*np.mean(NORM_STD)
    masque = NORM_STD > max_std

    # plt.suptitle(f"Algorithm with MBES\n{n_particles} particles; 1/{steps} data log used\nTotal time:{int(elapsed_time)}s")
    # ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    # ax2 = plt.subplot2grid((3, 2), (0, 1))
    # ax3 = plt.subplot2grid((3, 2), (1, 1))
    # ax4 = plt.subplot2grid((3, 2), (2, 1))
    #
    # print("Display the error and the final result..")
    # ax1.set_title("Barycentre")
    # ax1.set_xlabel("x [m]")
    # ax1.set_ylabel("y [m]")
    # ax1.plot(coord2cart((LAT,LON))[0,:], coord2cart((LAT,LON))[1,:],label='true position',linewidth=0.5,color='k')
    # scatter = ax1.scatter(BAR[:,0][~masque], BAR[:,1][~masque], s = 1.2, c = NORM_STD[~masque], cmap='plasma', label='particle cloud barycenter')
    # cbar = fig.colorbar(scatter, extend='both', ax = ax1)
    # cbar.set_label('Ecart type')
    # ax1.legend()
    #
    # ax2.set_title("Error function.")
    # ax2.set_xlabel("time [min]")
    # ax2.set_ylabel("error (m)")
    # # ax2.plot(TIME, ERR, color = 'b', label = 'erreur')
    # ax2.scatter(TIME, ERR, color = np.array(color_mbes), label = 'erreur', s = 1)
    # ax2.scatter(TIME, NORM_STD, color = 'green', label = 'ecart type', s = 1)
    # ERR = np.array(ERR)
    # idx_start = int(1/8*TIME.shape[0])
    # ax2.plot(TIME, np.mean(ERR)*np.ones(TIME.shape), label = f"mean error from beggining = {np.mean(ERR)}")
    # ax2.plot(TIME[idx_start:,], np.mean(ERR[idx_start:,])*np.ones(TIME[idx_start:,].shape), label = f"mean error from convergence = {np.mean(ERR[idx_start:,])}")
    # ax2.legend()
    #
    # X_gps, Y_gps = coord2cart((LAT, LON))
    # d_bottom_mnt = distance_to_bottom(np.column_stack((X_gps,Y_gps)),MNT)[1].squeeze()
    # mean_dvlR = (dvl_BM1R + dvl_BM2R + dvl_BM3R + dvl_BM4R)/4
    #
    # ax3.set_title("Different types of bottom measurements")
    # ax3.set_xlabel("Time [min]")
    # ax3.set_ylabel("Range [m]")
    # ax3.plot((dvl_T - dvl_T[0,])/60, mean_dvlR + offset_dvl, label = "z_dvl")
    # # ax3.plot((T - T[0,])/60, d_bottom_mnt, label = "z_mnt")
    # ax3.plot((MBES_mid_T - MBES_mid_T[0,])/60, MBES_mid_Z + offset_mbes, label = "z_mbes")
    # ax3.legend()
    #
    # ax4.set_title("Speed")
    # ax4.set_ylabel("v [m/s]")
    # ax4.set_xlabel("t [min]")
    # ax4.plot((dvl_T[steps:,] - dvl_T[steps,])/60, np.sqrt(dvl_v_x[steps:,]**2 + dvl_v_y[steps:,]**2), label = "dvl_speed")
    # ax4.plot(TIME, SPEED, label = "dvl_speed_filtered")
    # ax4.plot((T[steps:,] - T[steps,])/60, np.sqrt(V_X[steps:,]**2 + V_Y[steps:,]**2), label = "ins_speed")
    # ax4.legend()

    plt.suptitle(f"Algorithm with MBES\n{n_particles} particles; 1/{steps} data log used") #\nTotal time:{int(elapsed_time)}s")
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    idx_ti = int(T.shape[0]*0.6)
    idx_tf = int(T.shape[0]*0.75)
    print("Display the error and the final result..")
    ax1.set_title("Barycentre")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    # ax1.plot(coord2cart((LAT[idx_ti:idx_tf,],LON[idx_ti:idx_tf,]))[0,:], coord2cart((LAT[idx_ti:idx_tf,],LON[idx_ti:idx_tf,]))[1,:],label='true position',linewidth=0.5,color='k')
    # scatter = ax1.scatter(BAR[idx_ti:idx_tf,0][~masque], BAR[idx_ti:idx_tf,1][~masque], s = 1.2, c = NORM_STD[idx_ti:idx_tf,][~masque[idx_ti:idx_tf,]], cmap='plasma', label='particle cloud barycenter')
    ax1.plot(coord2cart((LAT[:,],LON[:,]))[0,:], coord2cart((LAT[:,],LON[:,]))[1,:],label='true position',linewidth=0.5,color='k')
    scatter = ax1.scatter(BAR[:,0][~masque], BAR[:,1][~masque], s = 1.2, c = NORM_STD[:,][~masque[:,]], cmap='plasma', label='particle cloud barycenter')
    cbar = fig.colorbar(scatter, extend='both', ax = ax1)
    cbar.set_label('Ecart type')
    ax1.legend()

    ax2.set_title("Error function.")
    ax2.set_xlabel("time [min]")
    ax2.set_ylabel("error (m)")
    # ax2.plot(TIME, ERR, color = 'b', label = 'erreur')
    ax2.scatter(TIME, ERR, color = 'blue', label = 'erreur', s = 1)
    ERR = np.array(ERR)
    idx_start = int(1/8*TIME.shape[0])
    ax2.plot(TIME, np.mean(ERR)*np.ones(TIME.shape), label = f"mean error from beggining = {np.mean(ERR)}")
    ax2.plot(TIME[idx_start:,], np.mean(ERR[idx_start:,])*np.ones(TIME[idx_start:,].shape), label = f"mean error from convergence = {np.mean(ERR[idx_start:,])}")
    ax2.legend()

    ax3.set_title("Std.")
    ax3.set_xlabel("time [min]")
    ax3.set_ylabel("std (m)")
    ax3.scatter(TIME, NORM_STD, color = 'blue', label = 'ecart type', s = 1)
    ax3.legend()


    print("Computing the diagrams..")

    plt.show()
    if bool_display:plt.pause(100)
