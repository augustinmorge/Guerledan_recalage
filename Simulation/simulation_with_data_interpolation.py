#!/usr/bin/env python3

n_particles = 100 #int(input("Number of particles: "))
steps = 1# int(input("number of steps between measures ? "))
bool_display = False #(str(input("Display the particles ? [Y/]"))=="Y")

import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
from resampler import Resampler
from storage.data_import import *
import sys
from tqdm import tqdm
file_path = os.path.dirname(os.path.abspath(__file__))

def sawtooth(x):
    return(2*np.arctan(np.tan(x/2)))

nx_mnt, ny_mnt = coord2cart((MNT[:,1],MNT[:,0]))
vec_mnt = np.vstack((nx_mnt, ny_mnt)).T

def interp_mnt(interp_points):
    _, ind = kd_tree.query(interp_points, k=4)
    ## Récupérer les valeurs z de chaque point MNT trouvé
    z_values = MNT[ind][:,:,2]

    # Calculer les coordonnées relatives de chaque point à interpoler par rapport au premier point MNT trouvé
    dx = (interp_points[:,0] - vec_mnt[ind[:,0]][:,0]) / (vec_mnt[ind[:,1]][:,0] - vec_mnt[ind[:,0]][:,0])
    dy = (interp_points[:,1] - vec_mnt[ind[:,0]][:,1]) / (vec_mnt[ind[:,2]][:,1] - vec_mnt[ind[:,0]][:,1])

    # Interpoler les valeurs z en utilisant l'interpolation bilinéaire
    z2 = z_values[:,0] + (z_values[:,1] - z_values[:,0]) * dx
    z3 = z_values[:,2] + (z_values[:,3] - z_values[:,2]) * dx
    z = z2 + (z3 - z2) * dy

    # Tableau des valeurs z interpolées
    interp_values = z

    ## Récupérer les valeurs z de chaque point MNT trouvé
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"interp_points={interp_points}")
    # print(f"vec_mnt[ind]={vec_mnt[ind]}")
    # print(f"vec_mnt[ind[:,0]]={vec_mnt[ind[:,0]]}")
    # print(f"z_values={z_values}")
    # print(f"z_values={z_values.shape}")
    # print(f"ind={ind}")

    # print(f"dist={dist}")
    # print(z2,z3,(z2-z3)*dy)
    # print(f"interp_points[:,0]={interp_points[:,0]}")
    # print(f"interp_points[:,1]={interp_points[:,1]}")
    # print(f"\nvec_mnt[ind]={vec_mnt[ind]}")
    # print(f"vec_mnt[ind[:,0]]={vec_mnt[ind[:,0]]}")
    # print(f"vec_mnt[ind[:,1]]={vec_mnt[ind[:,1]]}")
    # print(f"vec_mnt[ind[:,2]]={vec_mnt[ind[:,2]]}")
    # print(f"dx={dx}, dy={dy}")
    # print(f"interp_values={interp_values}")

    return(np.sqrt(dx**2+dy**2), np.expand_dims(interp_values,axis=1))



def initialize_particles_uniform(n_particles, bounds):
    particles = [1/n_particles*np.ones((n_particles,1)),[ \
                           np.random.uniform(bounds[0][0], bounds[0][1], n_particles).reshape(-1,1),
                           np.random.uniform(bounds[1][0], bounds[1][1], n_particles).reshape(-1,1)]]
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
    # weights[d_mnt > 10] = 0 # If we are out of the MNT
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

def compute_likelihood(samples, measurements, measurements_noise, beta):
    d_mnt, z_mbes_particule = interp_mnt(np.hstack((samples[1][0],samples[1][1])))

    # Map difference true and expected distance measurement to probability
    distance = np.abs(z_mbes_particule - measurements)
    # p_z_given_x_distance = np.exp(-beta*distance/(measurements_noise[0]**2))
    p_z_given_x_distance = np.exp(-beta*distance)

    # Return importance weight based on all landmarks
    return d_mnt, p_z_given_x_distance

def needs_resampling(resampling_threshold):
    return 1.0 / np.max(particles[0]) < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, measurements_noise, process_noise, particles,resampling_threshold, resampler, beta, bounds):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise, bounds)

    # Compute current particle's weight
    d_mnt, p = compute_likelihood(propagated_states, measurements, measurements_noise, beta)

    particules = validate_state(propagated_states, bounds, d_mnt)

    # Store
    propagated_states[0] = particles[0] * p

    # Update particles
    particles = normalize_weights(propagated_states)

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

    t_i = int(3/5*T.shape[0])
    t_f = int(4/5*T.shape[0]) #T.shape[0] #

    dt, t = set_dt(T[steps,], T[0,])
    _, tf = set_dt(T[t_f,])


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

        from PIL import Image
        image = Image.open("./storage/MNT_G1.png")
        image.show()

        print("Processing..")
        r = range(t_i,t_f,steps)


    else : r = tqdm(range(t_i,t_f,steps))
    fig, ax = plt.subplots()
    TIME = []; BAR = []; SPEED = []; ERR = []
    beta = 1/100.
    # beta = 1/steps

    for i in r:

        """Set data"""
        t = set_dt(T[i,])[1] #même dt pour tout t
        v_x = V_X[i,]
        v_y = V_Y[i,]
        v_z = V_Z[i,]
        lat = LAT[i,]
        lon = LON[i,]
        x_gps, y_gps = coord2cart((lat,lon)).flatten()
        d_mnt, measurements = interp_mnt(np.array([[x_gps, y_gps]]))
        # if d_mnt > 1:
        #     print(t, d_mnt)
        lat_std = LAT_STD[i,]
        lon_std = LON_STD[i,]
        v_x_std = V_X_STD[i,]
        v_y_std = V_Y_STD[i,]
        v_z_std = V_Z_STD[i,]

        """Processing the motion of the robot """
        robot_forward_motion =  dt*np.sqrt(v_x**2 + v_y**2)# + v_z**2)
        robot_angular_motion = np.arctan2(v_x,v_y) #Je sais pas pourquoi c'est à l'envers

        """ Processing error on measures"""
        meas_model_distance_std = None #1 #50*steps*(np.sqrt(lat_std**2 + lon_std**2)) # On estime que l'erreur en z est le même que celui en lat, lon, ce qui est faux
        measurements_noise = [meas_model_distance_std] ### Attention, std est en mètres !

        """ Processing error on algorithm"""
        motion_model_forward_std = steps*np.sqrt(v_y_std**2 + v_x_std**2)# + v_z_std**2)
        motion_model_turn_std = np.abs(sawtooth(np.arctan2((v_x + np.sign(v_x)*v_x_std),(v_y)) - np.arctan2((v_x),(v_y+np.sign(v_y)*v_y_std))))
        process_noise = [motion_model_forward_std, motion_model_turn_std]

        """Process the update"""
        t0 = time.time()
        particles = update(robot_forward_motion, robot_angular_motion, measurements,\
                           measurements_noise, process_noise, particles,\
                            resampling_threshold, resampler, beta = beta, bounds = bounds)

        """ Affichage en temps réel """
        if bool_display:
            ax.cla()
            print("Temps de calcul: ",time.time() - t0)
            t1 = time.time()
            ax.plot(coord2cart((LAT[t_i:t_f,], LON[t_i:t_f,]))[0,:], coord2cart((LAT[t_i:t_f,], LON[t_i:t_f,]))[1,:])
            ax.set_title("Particle filter with {} particles with z = {}m".format(n_particles, measurements))
            ax.set_xlim([x_gps_min - 100,x_gps_max + 100])
            ax.set_ylim([y_gps_min - 100,y_gps_max + 100])
            ax.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée', s = 100)
            ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles") # Affiche toutes les particules
            bx, by = get_average_state(particles)[0], get_average_state(particles)[1] #barycentre des particules
            ax.scatter(bx, by , color = 'green', label = 'Estimation of particles')

            ax.legend()

            # Redessin de la figure
            fig.canvas.draw()

            plt.pause(0.00001)
            print("Temps d'affichage: ",time.time()-t1,"\n")

        TIME.append(t)
        ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))
        BAR.append([get_average_state(particles)[0],get_average_state(particles)[1]])
        SPEED.append(np.sqrt(v_x**2 + v_y**2))# + v_z**2))
        # if test_diverge(ERR) : break #Permet de voir si l'algorithme diverge et pourquoi.

    elapsed_time = time.perf_counter() - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    """ Affichage final """
    BAR = np.array(BAR)
    LAT, LON = LAT[t_i:t_f,], LON[t_i:t_f,]

    plt.suptitle(f"Algorithm with interpolation\n{n_particles} particles\n1/{steps} data log used")
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
