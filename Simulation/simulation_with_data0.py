#!/usr/bin/env python3
import warnings
# Suppress all warning messages
warnings.filterwarnings('ignore')
import time
T_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
import copy
from resampler import Resampler
from npz.data_import import *
import PIL.Image as Image
import sys
from tqdm import tqdm
from matplotlib.patches import Ellipse
from ellipse_lib import *
from sklearn.neighbors import KDTree
from pyproj.transformer import transform
import pyproj
# Définit les coordonnées de référence
wpt_ponton = (48.1989495, -3.0148023)

def coord2cart(coords,coords_ref=wpt_ponton):
    R = 6372800
    ly,lx = coords
    lym,lxm = coords_ref
    x_tilde = R * np.cos(ly*np.pi/180)*(lx-lxm)*np.pi/180
    y_tilde = R * (ly-lym)*np.pi/180
    return np.array([x_tilde,y_tilde])

def cart2coord(pos_x, pos_y ,coords_ref=wpt_ponton):
    R = 6372800
    x,y = pos_x, pos_y
    lym,lxm = coords_ref
    ly = 180*y/(R*np.pi) + lym
    lx = lxm + 180*x/(np.pi*R*np.cos(ly*np.pi/180))
    return (ly,lx)


# from scipy.interpolate import griddata
#
# def distance_to_bottom(xy, mnt):
#     """
#     Given a reference MNT (x_mnt, y_mnt, z_mnt) of N points, interpolate Z at n * (xi, yi) coordinates.
#     :param xi: ndarray(n) of first coordinates
#     :param yi: ndarray(n) of second coordinates
#     :param x_mnt: ndarray(N) of first coordinates reference
#     :param y_mnt: ndarray(N) of second coordinates reference
#     :param z_mnt: ndarray(N) of measures reference
#     :return: zi: ndarray(n) of interpolated measures at points (xi, yi)
#     """
#     xi, yi = xy[:,0], xy[:,1]
#     points = np.hstack((mnt[:,0], mnt[:,1]))
#     z_mnt = mnt[:,2]
#     zi = griddata(points, z_mnt, (xi, yi), method='linear')  # method = linear / cubic / nearest
#     return zi

# About method choice for griddata :
# Nearest = nearest reference point gives the interpolated value : worst results
# Linear = use Delaunay triangulation to linearly interpolate data, no extrapolation out of reference bounds
# Cubic = cubic spline to interplate, higher time computation but often better results, extrapolate out of bounds

# x_mnt = MNT[:,0]
# y_mnt = MNT[:,1]
# gcs = pyproj.Proj(init='epsg:4326') # Define the WGS84 GCS
# proj = pyproj.Proj(init='epsg:2154') # Define the Lambert 93 projection
# lon_mnt, lat_mnt = pyproj.transform(proj, gcs, x_mnt, y_mnt) # Convert x and y values to latitude and longitude values
print("Building KDTree..")
lon_mnt, lat_mnt = MNT[:,0], MNT[:,1]
vec_mnt = np.vstack((lat_mnt, lon_mnt)).T
kd_tree = KDTree(vec_mnt, metric="euclidean")

def distance_to_bottom(xy,mnt):

    lat, lon = cart2coord(xy[:,0], xy[:,1])
    point = np.vstack((lat, lon)).T

    # Utilise KDTree pour calculer les distances
    distances, indices = kd_tree.query(point)
    # Récupère les altitudes des points les plus proches
    Z = mnt[indices,2]

    return Z

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

def compute_likelihood(samples, measurements, measurements_noise, beta):

    # meas_model_distance_std = 0.4
    # meas_model_angle_std = 0.3
    # measurements_noise = [meas_model_distance_std, meas_model_angle_std]

    # Map difference true and expected distance measurement to probability
    z_mbes_particule = distance_to_bottom(np.hstack((samples[1][0],samples[1][1])),MNT)
    # distance = np.abs(z_mbes_particule - measurements)
    distance = np.abs(z_mbes_particule - measurements)**2
    p_z_given_x_distance = np.exp(-beta*distance)

    # Return importance weight based on all landmarks
    return p_z_given_x_distance

def needs_resampling(resampling_threshold):

    max_weight = np.max(particles[0])

    return 1.0 / max_weight < resampling_threshold

def update(robot_forward_motion, robot_angular_motion, measurements, measurements_noise, process_noise, particles,resampling_threshold, resampler, beta):

    # Propagate the particle state according to the current particle
    propagated_states = propagate_sample(particles, robot_forward_motion, robot_angular_motion, process_noise)

    # Compute current particle's weight
    weights = particles[0] * compute_likelihood(propagated_states, measurements, measurements_noise, beta)

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

def set_dt(ti, pti = T[0,]):
    t = ti.split(":")
    t[2] = t[2].split(".")
    t = float(t[0])*60*60+float(t[1])*60+float(t[2][0]) + float(t[2][1])/1000.

    pt = pti.split(":")
    pt[2] = pt[2].split(".")
    pt = float(pt[0])*60*60+float(pt[1])*60+float(pt[2][0]) + float(pt[2][1])/1000.

    return(t - pt, t)

if __name__ == '__main__':
    bool_display = False
    for n_particles in [200,500,1000,2000]:

        x_gps_min, y_gps_min = np.min(coord2cart((LAT, LON))[0,:]), np.min(coord2cart((LAT, LON))[1,:])
        x_gps_max, y_gps_max = np.max(coord2cart((LAT, LON))[0,:]), np.max(coord2cart((LAT, LON))[1,:])
        bounds = [[x_gps_min, x_gps_max], [y_gps_min, y_gps_max]]
        particles = initialize_particles_uniform(n_particles, bounds)

        #For the update
        resampler = Resampler()
        resampling_threshold = 0.5*n_particles


        t_i = 0# T.shape[0]//2
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
            # r = range(t_i,t_f,steps)


        B = np.arange(0.001,2,0.05)
        S = [200,100,50,25,10,1]
        for steps in S:
            dt, t = set_dt(T[steps,], T[0,])
            r = tqdm(range(t_i,t_f,steps))
            M_ERR = []
            print(f"steps = {steps}")
            for beta in B:
                print(f"beta={beta}")
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
                    measurements = distance_to_bottom(np.array([[x_gps, y_gps]]), MNT)
                    lat_std = LAT_STD[i,]
                    lon_std = LON_STD[i,]
                    v_x_std = V_X_STD[i,]
                    v_y_std = V_Y_STD[i,]
                    v_z_std = V_Z_STD[i,]


                    """Processing the motion of the robot """
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
                    particles = update(robot_forward_motion, robot_angular_motion, measurements,\
                                       measurements_noise, process_noise, particles,\
                                        resampling_threshold, resampler, beta)

                    """ Affichage en temps réel """
                    if bool_display:
                        # if ERR != []:
                        #     if ERR[-1] > 40:
                        ax.cla()
                        print("Temps de calcul: ",time.time() - t0)
                        t1 = time.time()
                        ax.plot(coord2cart((LAT,LON))[0,:], coord2cart((LAT,LON))[1,:])
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

                    TIME.append(t)
                    ERR.append(np.sqrt((x_gps - get_average_state(particles)[0])**2 + (y_gps - get_average_state(particles)[1])**2))
                    BAR.append([get_average_state(particles)[0],get_average_state(particles)[1]])
                    SPEED.append(np.sqrt(v_x**2 + v_y**2 + v_z**2))
                    if test_diverge(ERR) : break #Permet de voir si l'algorithme diverge et pourquoi.

                M_ERR.append(np.mean(ERR))

            plt.title("variation of beta")
            plt.xlabel("beta")
            plt.ylabel("error")
            plt.plot(B, M_ERR)
            plt.savefig(f"imgs/test_error_n_particles{n_particles}_steps{steps}")
