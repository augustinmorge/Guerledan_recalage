# Simulation part
***
## Simulation on a lissajou

This script is a particle filter implementation in Python. The particle filter is a Monte Carlo method used to estimate the state of a system in the form of a probability distribution.

The input to the script includes the number of particles, the number of steps between measures, and a flag for whether to display the particles. The script then imports various libraries and defines several functions.

* **coord2cart** function converts coordinates to Cartesian coordinates using a reference point.
* **distance_to_bottom** function calculates the distance to the bottom and the altitude at the nearest point in a digital elevation model (DEM).
* **initialize_particles_uniform** function initializes particles with uniform weights within a specified bounds.
* **get_max_weight function** returns the maximum weight among the particles. The normalize_weights function normalizes the weights of the particles.
* **validate_state function** checks that the particles are within the bounds of the DEM and removes particles that are outside the bounds or too far from the ground.
* **propagate_sample** function propagates the particles according to a forward motion model and an angular motion model, with added process noise.
* **compute_importance** function calculates the importance weights of the particles based on their distance to the measured position.
* **resample_particles** function resamples the particles according to their importance weights.
* **plot_particles** function plots the particles on a map.

The script then reads the data for the digital elevation model (DEM) and generates a KD-tree for fast nearest neighbor search. It also reads the measurement data and converts the coordinates to Cartesian coordinates using the coord2cart function.

The script initializes the particles using the initialize_particles_uniform function and the bounds of the DEM. It then enters a loop to iterate through the measurement steps. At each step, it propagates the particles using the propagate_sample function, calculates the importance weights using the compute_importance function, resamples the particles using the resample_particles function, and plots the particles using the plot_particles function if the display flag is set to True.

At the end of the loop, the script prints the elapsed time and the estimated position.

## How the run the program ?
***
run `./simulation.py` for just a simuation with a lissajou curb
run `./simulation_with_data.py`for the simulation with our data
run `./simulation_with_data_\&beams.py`for a simulation with data using beams
