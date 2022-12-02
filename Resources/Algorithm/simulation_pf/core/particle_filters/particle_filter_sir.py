from .particle_filter_base import ParticleFilter
from core.resampling.resampler import Resampler


class ParticleFilterSIR(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * propagation and measurement models are largely hardcoded (except for standard deviations.
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm,
                 resampling_threshold):
        """
        Initialize the SIR particle filter.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        """
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampling_threshold = resampling_threshold
        self.resampler = Resampler()

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Propagate the particle state according to the current particle
        # propagated_state = self.propagate_samples(self.particles[1:], robot_forward_motion, robot_angular_motion)
        propagated_state = self.propagate_samples(self.particles[1])

        # Compute current particle's weight
        weight = self.particles[0] * self.compute_likelihood(propagated_state, measurements, landmarks)

        # Store
        # new_particles.append([weight, propagated_state])
        new_particles = [weight, propagated_state[0], propagated_state[1], propagated_state[2]]


        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        if 1.0 / np.sum(self.particles[0]**2) < self.resampling_threshold:
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
