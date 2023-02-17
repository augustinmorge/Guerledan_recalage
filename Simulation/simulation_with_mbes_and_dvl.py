#!/usr/bin/env python3

# from storage.data_import import *
# offset_dvl = -115.5714023521081
# offset_mbes = -117.6155899936386
# from storage_afternoon.data_import import *
# offset_dvl = -116.48084912914656
# offset_mbes = -117.67756491403492
from storage_final.data_import import *
offset_dvl = 119.91869636276917
offset_mbes = 2.2981554769660306
n_particles = int(input("Number of particles: "))
steps = int(input("Number of steps between measures ? "))
bool_display = (str(input("Display the particles ? [Y/]"))=="Y")
using_offset = True # str(input("Using offset ? [Y/]")) == "Y"

ct_resampling = 0

import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
from simulation_helper.resampler import Resampler
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
    if using_offset : d_MBES_mid_particule = new_z_particules_mnt
    else : d_MBES_mid_particule = new_z_particules_mnt - z_particules_mnt

    mbes_min_Z, mbes_mid_Z, mbes_max_Z, \
        dp_x_mid, dp_y_mid, dp_x_min, dp_y_min, dp_x_max, dp_y_max = \
            measurements.flatten()

    _, d_mbes_particule_min = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_min,propagated_states[1][1]+dp_y_min)),MNT)
    _, d_mbes_particule_mid = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_mid,propagated_states[1][1]+dp_y_mid)),MNT)
    # _, d_mbes_particule_mid = distance_to_bottom(np.hstack((propagated_states[1][0],propagated_states[1][1])),MNT)
    _, d_mbes_particule_max = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_max,propagated_states[1][1]+dp_y_max)),MNT)

    # Map difference true and expected distance measurement to probability
    # distance = np.abs(d_MBES_mid_particule-measurements)
    distance_min = np.abs(mbes_min_Z - d_mbes_particule_min)
    distance_mid = np.abs(mbes_mid_Z - d_mbes_particule_mid)
    distance_max = np.abs(mbes_max_Z - d_mbes_particule_max)

    if measurements_noise[0] == None:
        # p_z_given_x_distance = np.exp(-beta*distance_min)*np.exp(-beta*distance_mid**2)*np.exp(-beta*distance_max)
        # p_z_given_x_distance = np.exp(-beta*distance_min)*np.exp(-beta*distance_mid**2)*np.exp(-beta*distance_max)
        p_z_given_x_distance = np.exp(-beta*distance_mid**2)

    else:
        p_z_given_x_distance = np.exp(-beta*distance/(measurements_noise[0]**2))

    # p_z_given_x_distance = 1
    # Return importance weight based on all landmarkss

    if dtmbes == 0: #si il n'y a pas de nouvelle données MBES
        dvl_bm1r, dvl_bm2r, dvl_bm3r, dvl_bm4r = [measurements_dvl[i] for i in range(4)]
        use_4_beams = False
        if use_4_beams:
            th = np.pi/4
            pi2_janus = 60*np.pi/180
            opp_angle = np.tan(60*np.pi/180)

            dp_x_B1 = -dvl_bm1r/opp_angle*np.sin(yaw-th)
            dp_y_B1 = dvl_bm1r/opp_angle*np.cos(yaw-th)

            dp_x_B2 = dvl_bm2r/opp_angle*np.sin(yaw-th)
            dp_y_B2 = -dvl_bm2r/opp_angle*np.cos(yaw-th)

            dp_x_B3 = dvl_bm3r/opp_angle*np.cos(yaw-th)
            dp_y_B3 = dvl_bm3r/opp_angle*np.sin(yaw-th)

            dp_x_B4 = -dvl_bm4r/opp_angle*np.cos(yaw-th)
            dp_y_B4 = -dvl_bm4r/opp_angle*np.sin(yaw-th)

            d_mnt_B1, d_mbes_particule_B1 = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_B1,propagated_states[1][1]+dp_y_B1)),MNT)
            d_mnt_B2, d_mbes_particule_B2 = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_B2,propagated_states[1][1]+dp_y_B2)),MNT)
            d_mnt_B3, d_mbes_particule_B3 = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_B3,propagated_states[1][1]+dp_y_B3)),MNT)
            d_mnt_B4, d_mbes_particule_B4 = distance_to_bottom(np.hstack((propagated_states[1][0]+dp_x_B4,propagated_states[1][1]+dp_y_B4)),MNT)

            # Map difference true and expected distance measurement to probability
            distance_B1 = np.abs(d_mbes_particule_B1-dvl_bm1r)
            distance_B2 = np.abs(d_mbes_particule_B2-dvl_bm2r)
            distance_B3 = np.abs(d_mbes_particule_B3-dvl_bm3r)
            distance_B4 = np.abs(d_mbes_particule_B4-dvl_bm4r)
            # print('distance : ', distance_B2[127])


            # print('distances : ', distance_B1[5], distance_B2[5], distance_B3[5], distance_B4[5])
            # print('d_mbes_particules : ', d_mbes_particule_B1[5], d_mbes_particule_B2[5], d_mbes_particule_B3[5], d_mbes_particule_B4[5])
            # print('dvl_bmRr: ', dvl_bm1r, dvl_bm2r, dvl_bm3r, dvl_bm4r)

            if measurements_noise[0] == None:
                if use_4_beams : #1000 1/4
                    # p_z_given_x_distance = (np.exp(-beta/100*pow(distance_B3,1/4))+np.exp(-beta/100*pow(distance_B4,1/4))+np.exp(-beta/100*pow(distance_B2,1/4))+np.exp(-beta/100*pow(distance_B1,1/4)))/4
                    p_z_given_x_distance = (np.exp(-beta/300*pow(distance_B3,1))*np.exp(-beta/300*pow(distance_B4,1))*np.exp(-beta/300*pow(distance_B2,1))*np.exp(-beta/300*pow(distance_B1,1)))
                # p1, p2, p3, p4 = np.exp(-beta*distance_B1**2), np.exp(-beta*distance_B2**2), np.exp(-beta*distance_B3**2), np.exp(-beta*distance_B4**2)
            else:
                p_z_given_x_distance = np.exp(-beta*distance_B1/(measurements_noise[0]**2))*np.exp(-beta*distance_B2/(measurements_noise[0]**2))*np.exp(-beta*distance_B3/(measurements_noise[0]**2))*np.exp(-beta*distance_B4/(measurements_noise[0]**2))
            #
            # d_mnt = np.array([d_mnt_B1, d_mnt_B2, d_mnt_B3, d_mnt_B4])
            # new_z_particules_mnt = np.array([d_mbes_particule_B1, d_mbes_particule_B2, d_mbes_particule_B3, d_mbes_particule_B4])

        else:
            if measurements_noise[0] == None:
                mean_range_dvl = (dvl_bm1r*dvl_bm2r)/(dvl_bm1r+dvl_bm2r) + (dvl_bm3r*dvl_bm4r)/(dvl_bm3r+dvl_bm4r)
                d = np.abs(new_z_particules_mnt - mean_range_dvl)
                p_z_given_x_distance = np.exp(-beta*d**2)
            else:
                p_z_given_x_distance = np.exp(-beta*d/(measurements_noise[0]**2))

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
    avg_x = np.sum(particles[0]*particles[1][0])
    avg_y = np.sum(particles[0]*particles[1][1])

    return [avg_x, avg_y]

def get_std_state(particles):

    # Compute weighted average
    std_x = np.sqrt(np.sum(particles[0]*particles[1][0]**2)/n_particles - (np.sum(particles[0]*particles[1][0])/n_particles)**2) # / np.sum(particles[0])
    std_y = np.sqrt(np.sum(particles[0]*particles[1][1]**2)/n_particles - (np.sum(particles[0]*particles[1][1])/n_particles)**2) # / np.sum(particles[0])

    return std_x, std_y

# Init range sensor
previous_measurements = MBES_mid_Z[0,]

def f_measurements(i, previous_measurements):
    measurements = MBES_mid_Z[i,] - previous_measurements #117.61492204 #
    return measurements, MBES_mid_Z[i,], None

ct_mbes = 0; dtmbes = 0.15
color_mbes = []
def f_measurements_offset(i):
    global ct_mbes
    previous_ct_mbes = ct_mbes
    while MBES_mid_T[ct_mbes,] <= T[i,]:
        ct_mbes += 1
    # measurements = MBES_mid_Z[ct_mbes,]

    global dtmbes, color_mbes
    dtmbes = MBES_mid_T[ct_mbes,] - MBES_mid_T[previous_ct_mbes,]
    if dtmbes == 0:
        color_mbes.append('red')
    else:
        color_mbes.append('blue')

    angle_mbes = 62.5
    angle_max = (90 - (angle_mbes - (256 - MBES_max_idx[ct_mbes,])*angle_mbes/128))
    angle_min = -(90 - (angle_mbes - (MBES_min_idx[ct_mbes,] - 1)*angle_mbes/128))
    angle_mid = 90 - (128 - MBES_mid_idx[ct_mbes,])*angle_mbes/128

    dp_x_mid = MBES_mid_Z[ct_mbes,]/np.tan(angle_mid*np.pi/180)*np.cos(YAW[i,])
    dp_y_mid = MBES_mid_Z[ct_mbes,]/np.tan(angle_mid*np.pi/180)*np.sin(YAW[i,])

    dp_x_min = -MBES_min_Z[ct_mbes,]/np.tan(angle_min*np.pi/180)*np.cos(YAW[i,]-3*np.pi/2)
    dp_y_min = -MBES_min_Z[ct_mbes,]/np.tan(angle_min*np.pi/180)*np.sin(YAW[i,]-3*np.pi/2)

    dp_x_max = MBES_max_Z[ct_mbes,]/np.tan(angle_max*np.pi/180)*np.cos(YAW[i,]-np.pi/2)
    dp_y_max = MBES_max_Z[ct_mbes,]/np.tan(angle_max*np.pi/180)*np.sin(YAW[i,]-np.pi/2)

    measurements = np.array([MBES_min_Z[ct_mbes,] + offset_mbes, MBES_mid_Z[ct_mbes,] + offset_mbes, MBES_max_Z[ct_mbes,] + offset_mbes, \
                            dp_x_mid, dp_y_mid, dp_x_min, dp_y_min, dp_x_max, dp_y_max])

    return measurements, None

def f_measurements_offset_dvl(i):
    # range_dvl = np.array([dvl_BM1R[i,], dvl_BM2R[i,], dvl_BM3R[i,], dvl_BM4R[i,]])
    # dvl_BM1R[i,], dvl_BM2R[i,], dvl_BM3R[i,], dvl_BM4R[i,] = \
    #     filter_lpf_dvlr.low_pass_next(np.array([dvl_BM1R[i,], dvl_BM2R[i,], dvl_BM3R[i,], dvl_BM4R[i,]])).flatten()
    range_dvl = np.array([dvl_BM1R[i,], dvl_BM2R[i,], dvl_BM3R[i,], dvl_BM4R[i,]])
    measurements = range_dvl - offset_dvl
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
    # resampling_threshold = 1/2*n_particles
    idx_ti = 0
    idx_tf =  dvl_T.shape[0]

    dt = dvl_T[steps,] - dvl_T[0,]
    tini = dvl_T[idx_ti,]

    if bool_display:
        """ Création des isobath """
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
    # beta = 10**(-1.37)
    beta = 0.1
    filter_lpf_speed = Low_pass_filter(1., np.array([dvl_v_x[0,], dvl_v_y[0,]]))

    for i in r:

        """Set data"""
        #Use the DVL
        t = dvl_T[i,]
        yaw = YAW[i,]
        yaw_std = YAW_STD[i,]
        v_x, v_y = filter_lpf_speed.low_pass_next(np.array([dvl_v_x[i,], dvl_v_y[i,]])).flatten()
        # v_std = dvl_VSTD[i,]
        # v_std = 0.4*10*dt_br
        v_std = 0.4*10*dt_br

        #Use the INS
        # t = T[i,]
        # yaw = YAW[i,]
        # # yaw_std = YAW_STD[i,]
        # yaw_std = np.abs(np.arctan2(V_Y[i,], V_X[i,] + V_X_STD[i,]) - np.arctan2(V_Y[i,] + V_Y_STD[i,], V_X[i,]))
        # v_x, v_y = V_X[i,], V_Y[i,]
        # v_std = dt*np.sqrt(V_X_STD[i,]**2 + V_Y_STD[i,]**2)

        if using_offset :
            measurements, meas_model_distance_std = f_measurements_offset(i)
            measurements_dvl, meas_model_distance_std_dvl = f_measurements_offset_dvl(i)
        else: measurements, previous_measurements, meas_model_distance_std = f_measurements(i, previous_measurements)

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
            ax.set_title("Particle filter with {} particles".format(n_particles))# with z = {}m".format(n_particles, measurements))
            ax.set_xlim([x_gps_min - 100,x_gps_max + 100])
            ax.set_ylim([y_gps_min - 100,y_gps_max + 100])
            bx, by = get_average_state(particles)[0], get_average_state(particles)[1] #barycentre des particules
            scatter1 = ax.scatter(x_gps, y_gps ,color='blue', label = 'True position panopée', s = 100)
            scatter2 = ax.scatter(particles[1][0], particles[1][1], color = 'red', s = 0.8, label = "particles",alpha=particles[0][:,0]/pow(np.max(particles[0][:,0]),2/3))
            scatter3 = ax.scatter(bx, by , color = 'green', label = 'Estimation of particles')

            if dtmbes == 0:
                plt.plot([], [], marker='o', color='red', label='MBES: off', markerfacecolor='red', markersize=10)
                plt.plot([], [], marker='o', color='green', label='DVLbr: on', markerfacecolor='red', markersize=10)
            else:
                plt.plot([], [], marker='o', color='green', label='MBES: on', markerfacecolor='green', markersize=10)
                plt.plot([], [], marker='o', color='red', label='DVLbr: off', markerfacecolor='green', markersize=10)

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
    max_std = 3*np.mean(NORM_STD)
    masque = NORM_STD > max_std

    plt.suptitle(f"Algorithm with MBES\n{n_particles} particles; 1/{steps} data log used\nTotal time:{int(elapsed_time)}s")
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
    # ax2.plot(TIME, ERR, color = 'b', label = 'erreur')
    ax2.scatter(TIME, ERR, color = np.array(color_mbes), label = 'erreur', s = 1)
    ax2.scatter(TIME, NORM_STD, color = 'green', label = 'ecart type', s = 1)
    ERR = np.array(ERR)
    idx_start = int(1/8*TIME.shape[0])
    ax2.plot(TIME, np.mean(ERR)*np.ones(TIME.shape), label = f"mean error from beggining = {np.mean(ERR)}")
    ax2.plot(TIME[idx_start:,], np.mean(ERR[idx_start:,])*np.ones(TIME[idx_start:,].shape), label = f"mean error from convergence = {np.mean(ERR[idx_start:,])}")
    ax2.legend()

    X_gps, Y_gps = coord2cart((LAT, LON))
    d_bottom_mnt = distance_to_bottom(np.column_stack((X_gps,Y_gps)),MNT)[1].squeeze()
    mean_dvlR = (dvl_BM1R + dvl_BM2R + dvl_BM3R + dvl_BM4R)/4

    ax3.set_title("Different types of bottom measurements")
    ax3.set_xlabel("Time [min]")
    ax3.set_ylabel("Range [m]")
    ax3.plot((dvl_T - dvl_T[0,])/60, mean_dvlR - offset_dvl, label = "z_dvl")
    ax3.plot((T - T[0,])/60, d_bottom_mnt, label = "z_mnt")
    ax3.plot((MBES_mid_T - MBES_mid_T[0,])/60, MBES_mid_Z + offset_mbes, label = "z_mbes")
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
