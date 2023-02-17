# Simulation part
***
## First steps
### Logs
We did several tests :
  * The 13/10/22 at 1pm. (G1)
  * The 13/10/22 at 3pm. (G1)
  * The 09/02/23 at 11am. (G2)
  * The 09/02/23 at 1pm. (G2)

You have to download the .txt file from :
  * [G1: 1pm](https://mega.nz/folder/LddXFIjJ#8aNVKljeaiCF3S-_OeuZqg) in /storage
  * [G1: 3pm](https://mega.nz/folder/6JMSFIoS#Je4uvVFECIoUeqyPqkgzfQ) in /storage_afternoon
  * [G2: 11am](https://mega.nz/folder/fFsSGLoT#b_4goBnMMQjKI6HvGNJZ5w) in storage_semi_final
  * [G2: 1pm](https://mega.nz/folder/HMVAVaib#aon7IKUUBNBkaCgNgTzOxg) in storage_final


Then go into the right storage folder and run /data_import.py with **1** in bool_txt and **0** in bool_compress. Then relaunch ./data_import.py with the opposite.

Once it's done you can look at your map : `./display_mnt.py` or run the simulation.

This is the trajectory and the DTM of the first test for example:

<div style="text-align:center">
<p align="center">
<img src="https://github.com/augustinmorge/Guerledan_recalage/blob/main/Simulation/storage/MNT_G1.png" width="300" title="DVL : Pathfinder">
</p>
</div>


## Simulations

### Explainations

Those script is a particle filter implementation in Python. The particle filter is a Monte Carlo method used to estimate the state of a system in the form of a probability distribution.

The input to the script includes the number of particles, the number of steps between measures, and a flag for whether to display the particles. The script then imports various libraries and defines several functions.

* **coord2cart** converts coordinates to Cartesian coordinates using a reference point.
* **distance_to_bottom** calculates the distance to the bottom and the altitude at the nearest point in a digital elevation model (DEM).
* **initialize_particles_uniform** initializes particles with uniform weights within a specified bounds.
* **get_max_weight** returns the maximum weight among the particles.
* **normalize_weights** normalizes the weights of the particles.
* **validate_state** checks that the particles are within the bounds of the DEM and removes particles that are outside the bounds or too far from the ground.
* **propagate_sample** propagates the particles according to a forward motion model and an angular motion model, with added process noise.
* **compute_importance** calculates the importance weights of the particles based on their distance to the measured position.
* **resample_particles** resamples the particles according to their importance weights.

The script then reads the data for the digital elevation model (DEM) and generates a KD-tree for fast nearest neighbor search. It also reads the measurement data and converts the coordinates to Cartesian coordinates using the coord2cart function.
The script initializes the particles using the initialize_particles_uniform function and the bounds of the DEM. It then enters a loop to iterate through the measurement steps. At each step, it propagates the particles using the propagate_sample function, calculates the importance weights using the compute_importance function, resamples the particles using the resample_particles function, and plots the particles using the plot_particles function if the display flag is set to True.

The script then enters a loop where it performs the following steps at each iteration:

1. Propagates the particles using the propagate_sample function.
2. Calculates the distance and altitude of each particle using the distance_to_bottom function.
3. Calculates the likelihood of each particle using the distance and altitude information.
4. Resamples the particles using the Resampler class.
5. Normalizes the weights of the resampled particles using the normalize_weights function.
6. Validates the state of the particles using the validate_state function.
7. If the display flag is set, plots the particles and waypoints on a map.

At the end of the loop, the script prints the elapsed time and the estimated position.

### How the run the program ?
***
run `./simulation.py` for just a simuation with a lissajou curb

run `./simulation_with_data.py`for the simulation with our data

## Folders
  * *storage*: store the logs of the tests and create compressed files inside (npz, joblib,..)
  * *mnt* : The DTM used for the tests. It's the one by EDF in 2013. It has its own folder because we use the same for every tests.
